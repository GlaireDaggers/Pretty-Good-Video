use std::io::{Read, Write};

use byteorder::ReadBytesExt;

#[derive(Debug)]
pub enum HuffmanError {
    EncodeError,
    DecodeError,
    IOError(std::io::Error),
}

pub struct HuffmanTree {
    codes: [Code;256],
    table: [u8;256],
    dec_table: [Code;256],
    root: Box<Node>,
}

struct BitReader<R: Read> {
    buf: Vec<u8>,
    buf_pos: usize,
    reader: R,
}

struct BitWriter<W: Write> {
    buf: Vec<u8>,
    buf_pos: usize,
    writer: W,
}

impl<R: Read> BitReader<R> {
    pub fn new(reader: R) -> BitReader<R> {
        BitReader { buf: Vec::new(), buf_pos: 0, reader: reader }
    }

    pub fn read_1(self: &mut BitReader<R>) -> Result<bool, std::io::Error> {
        let buf_cursor = self.buf_pos / 8;

        while self.buf.len() <= buf_cursor {
            self.buf.push(match self.reader.read_u8() {
                Ok(v) => v,
                Err(e) => match e.kind() {
                    std::io::ErrorKind::UnexpectedEof => {
                        0
                    }
                    _ => {
                        return Err(e);
                    }
                }
            });
        }

        let bit_pos = self.buf_pos % 8;
        let cur = self.buf[buf_cursor] & (1 << bit_pos) != 0;

        self.buf_pos += 1;
        Ok(cur)
    }

    pub fn read_8(self: &mut BitReader<R>) -> Result<u8, std::io::Error> {
        let buf_cursor = self.buf_pos / 8;

        while self.buf.len() <= buf_cursor + 1 {
            self.buf.push(match self.reader.read_u8() {
                Ok(v) => v,
                Err(e) => match e.kind() {
                    std::io::ErrorKind::UnexpectedEof => {
                        0
                    }
                    _ => {
                        return Err(e);
                    }
                }
            });
        }

        let bit_pos = self.buf_pos % 8;
        let cur = self.buf[buf_cursor] >> bit_pos |
            self.buf[buf_cursor + 1].checked_shl(8 - bit_pos as u32).unwrap_or(0);

        self.buf_pos += 8;
        Ok(cur)
    }

    pub fn rewind(self: &mut BitReader<R>, bits: usize) {
        self.buf_pos -= bits;
    }
}

impl<W: Write> BitWriter<W> {
    pub fn new(writer: W) -> BitWriter<W> {
        BitWriter { buf: Vec::new(), buf_pos: 0, writer: writer }
    }

    pub fn write_1(self: &mut BitWriter<W>, bit: bool) {
        let buf_cursor = self.buf_pos / 8;

        while self.buf.len() <= buf_cursor {
            self.buf.push(0);
        }

        let bit_pos = self.buf_pos % 8;
        self.buf[buf_cursor] |= (bit as u8) << bit_pos;
        self.buf_pos += 1;
    }

    pub fn write_bits(self: &mut BitWriter<W>, mut val: u32, mut bits: usize) {
        while bits > 0 {
            self.write_1(val & 1 != 0);
            val >>= 1;
            bits -= 1;
        }
    }

    pub fn flush(self: &mut BitWriter<W>) -> Result<(), std::io::Error> {
        self.writer.write_all(&self.buf)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Code {
    val: u32,
    len: u32,
    symbol: u8,
}

impl Code {
    pub fn new() -> Code {
        Code { val: 0, len: 0, symbol: 0 }
    }

    pub fn append(self: &Code, bit: bool) -> Code {
        Code { val: self.val | ((bit as u32) << self.len), len: self.len + 1, symbol: self.symbol }
    }

    pub fn mask(self: &Code) -> u32 {
        (1 << self.len) - 1
    }
}

struct Node {
    freq: u32,
    ch: Option<u8>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    pub fn new(freq: u32, ch: Option<u8>) -> Node {
        Node { freq: freq, ch: ch, left: None, right: None }
    }

    pub fn into_box(self: Node) -> Box<Node> {
        Box::new(self)
    }
}

impl HuffmanTree {
    pub fn empty() -> HuffmanTree {
        HuffmanTree { codes: [Code::new();256], table: [0;256], dec_table: [Code::new();256], root: Node::new(0, None).into_box() }
    }

    pub fn from_data(data: &[u8]) -> HuffmanTree {
        let freq = create_table(data);
        HuffmanTree::from_table(&freq)
    }

    pub fn from_table(table: &[u8;256]) -> HuffmanTree {
        let mut p:Vec<Box<Node>> = Vec::new();

        for (ch, fr) in table.iter().enumerate() {
            if *fr > 0 {
                p.push(Node::new(*fr as u32, Some(ch as u8)).into_box());
            }
        }

        while p.len() > 1 {
            p.sort_by(|a, b| (&(b.freq)).cmp(&(a.freq)));
            let a = p.pop().unwrap();
            let b = p.pop().unwrap();
            let mut c = Node::new(a.freq + b.freq, None).into_box();
            c.left = Some(a);
            c.right = Some(b);
            p.push(c);
        }

        assert!(p.len() > 0);

        let root = p.pop().unwrap();

        let mut codes = [Code::new();256];
        assign_codes(&root, &mut codes, Code::new());

        // generate pre-masked decoder table for codes of 8 bits or less - allows us to just read in a whole u8 and index into this table to get a code
        // if a code is longer than 8 bits, it falls back to the slow tree traversal path

        let mut dec_table = [Code::new();256];

        for val in 0..256 {
            for c in codes {
                if c.len > 0 && c.len <= 8 && val & c.mask() == c.val {
                    dec_table[val as usize] = c;
                    break;
                }
            }
        }

        HuffmanTree { codes: codes, table: table.clone(), dec_table: dec_table, root: root }
    }

    pub fn get_table(self: &HuffmanTree) -> &[u8;256] {
        &self.table
    }

    fn read_slow<R: Read>(self: &HuffmanTree, bitreader: &mut BitReader<R>) -> Result<u8, HuffmanError> {
        let mut node = &self.root;

        loop {
            if let Some(ch) = node.ch {
                return Ok(ch);
            } else {
                let bit = match bitreader.read_1() {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(HuffmanError::IOError(e));
                    }
                };

                if bit {
                    if let Some(ref r) = node.right {
                        node = r;
                    } else {
                        return Err(HuffmanError::DecodeError);
                    }
                } else {
                    if let Some(ref l) = node.left {
                        node = l;
                    } else {
                        return Err(HuffmanError::DecodeError);
                    }
                }
            }
        }
    }

    pub fn read<R: Read>(self: &HuffmanTree, reader: &mut R, buffer: &mut [u8]) -> Result<(), HuffmanError> {
        let mut bitreader = BitReader::new(reader);

        for idx in 0..buffer.len() {
            let cur = match bitreader.read_8() {
                Ok(v) => v,
                Err(e) => {
                    return Err(HuffmanError::IOError(e));
                }
            };

            let c = self.dec_table[cur as usize];
            if c.len == 0 {
                // couldn't find code in fast table, try slow lookup instead
                bitreader.rewind(8);
                buffer[idx] = self.read_slow(&mut bitreader)?;
            } else {
                buffer[idx] = c.symbol;
                bitreader.rewind(8 - c.len as usize);
            }
        }

        Ok(())
    }

    pub fn write<W: Write>(self: &HuffmanTree, writer: &mut W, data: &[u8]) -> Result<(), HuffmanError> {
        let mut bitwriter = BitWriter::new(writer);

        for idx in 0..data.len() {
            let val = data[idx];
            let c = self.codes[val as usize];
            if c.len == 0 {
                return Err(HuffmanError::EncodeError);
            }

            bitwriter.write_bits(c.val, c.len as usize);
        }

        match bitwriter.flush() {
            Ok(_) => {}
            Err(e) => {
                return Err(HuffmanError::IOError(e));
            }
        };

        Ok(())
    }
}

fn assign_codes(p: &Box<Node>, h: &mut [Code;256], s: Code) {
    if let Some(ch) = p.ch {
        let s = Code { val: s.val, len: s.len, symbol: ch };
        h[ch as usize] = s;
    } else {
        if let Some(ref l) = p.left {
            assign_codes(l, h, s.append(false));
        }

        if let Some(ref r) = p.right {
            assign_codes(r, h, s.append(true));
        }
    }
}

fn create_table(data: &[u8]) -> [u8;256] {
    let mut freq_table: [i32;256] = [0;256];
    let mut max = 0;

    for d in data {
        freq_table[*d as usize] += 1;
        if freq_table[*d as usize] > max {
            max = freq_table[*d as usize];
        }
    }

    // normalize to u8 range

    let mut normalized_data: [u8; 256] = [0;256];

    for i in 0..freq_table.len() {
        if freq_table[i] > 0 {
            normalized_data[i] = (freq_table[i] * 255 / max).max(1) as u8;
        }
    }

    normalized_data
}