#ifndef _PGV_H
#define _PGV_H

#include <stdint.h>

typedef struct PGV_Stream {
    void *context;
    size_t read_fn(void *context, uint8_t *buffer, size_t bufferlen);
    uint64_t seek_fn(void *context, int64_t offset, int32_t whence);
} PGV_Stream;

typedef void PGV_Decoder;

extern PGV_Decoder *pgv_decoder_new(PGV_Stream *stream);
extern void pgv_decoder_destroy(PGV_Decoder *decoder);

extern uint32_t pgv_decoder_width(PGV_Decoder *decoder);
extern uint32_t pgv_decoder_height(PGV_Decoder *decoder);
extern uint32_t pgv_decoder_num_frames(PGV_Decoder *decoder);
extern uint32_t pgv_decoder_framerate(PGV_Decoder *decoder);
extern uint32_t pgv_decoder_num_audio_frames(PGV_Decoder *decoder);
extern uint32_t pgv_decoder_audio_channels(PGV_Decoder *decoder);
extern uint32_t pgv_decoder_audio_samplerate(PGV_Decoder *decoder);
extern uint32_t pgv_decoder_audio_sync_hint(PGV_Decoder *decoder);

extern int32_t pgv_decoder_decode_frame(PGV_Decoder *decoder, uint8_t *buf_y, uint8_t *buf_u, uint8_t *buf_v);
extern int32_t pgv_decoder_decode_audio(PGV_Decoder *decoder, int16_t **buf, size_t samples);

#endif