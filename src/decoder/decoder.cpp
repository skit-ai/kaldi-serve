// decoder-cpu.cpp - CPU Decoder Implementation

// local includes
#include "config.hpp"
#include "decoder.hpp"
#include "types.hpp"


namespace kaldiserve {

Decoder::Decoder(ChainModel *const model) : model_(model) {

    if (model_->wb_info != nullptr) options.enable_word_level = true;
    if (model_->rnnlm_info != nullptr) options.enable_rnnlm = true;

    // decoder vars initialization
    decoder_ = NULL;
    feature_pipeline_ = NULL;
    silence_weighting_ = NULL;
    adaptation_state_ = NULL;
}

Decoder::~Decoder() noexcept {
    free_decoder();
}

void Decoder::start_decoding(const std::string &uuid) noexcept {
    free_decoder();

    adaptation_state_ = new kaldi::OnlineIvectorExtractorAdaptationState(model_->feature_info->ivector_extractor_info);

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline(*model_->feature_info);
    feature_pipeline_->SetAdaptationState(*adaptation_state_);

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->lattice_faster_decoder_config,
                                                      model_->trans_model, *model_->decodable_info,
                                                      *model_->decode_fst, feature_pipeline_);
    decoder_->InitDecoding();

    silence_weighting_ = new kaldi::OnlineSilenceWeighting(model_->trans_model,
                                                           model_->feature_info->silence_weighting_config,
                                                           model_->decodable_opts.frame_subsampling_factor);

    uuid_ = uuid;
}

void Decoder::free_decoder() noexcept {
    if (decoder_) {
        delete decoder_;
        decoder_ = NULL;
    }
    if (adaptation_state_) {
        delete adaptation_state_;
        adaptation_state_ = NULL;
    }
    if (feature_pipeline_) {
        delete feature_pipeline_; 
        feature_pipeline_ = NULL;
    }
    if (silence_weighting_) {
        delete silence_weighting_;
        silence_weighting_ = NULL;
    }
    uuid_ = "";
}

void Decoder::decode_stream_wav_chunk(std::istream &wav_stream) {
    kaldi::WaveData wave_data;
    wave_data.Read(wav_stream);

    const kaldi::BaseFloat samp_freq = wave_data.SampFreq();

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> wave_part(wave_data.Data(), 0);
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;
    _decode_wave(wave_part, delta_weights, samp_freq);
}

void Decoder::decode_stream_raw_wav_chunk(std::istream &wav_stream,
                                          const float& samp_freq,
                                          const int &data_bytes) {
    kaldi::Matrix<kaldi::BaseFloat> wave_matrix;    
    read_raw_wav_stream(wav_stream, data_bytes, wave_matrix);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> wave_part(wave_matrix, 0);
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;

    _decode_wave(wave_part, delta_weights, samp_freq);
}

void Decoder::decode_wav_audio(std::istream &wav_stream,
                               const float &chunk_size) {
    kaldi::WaveData wave_data;
    wave_data.Read(wav_stream);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);
    const kaldi::BaseFloat samp_freq = wave_data.SampFreq();

    int32 chunk_length;
    if (chunk_size > 0) {
        chunk_length = int32(samp_freq * chunk_size);
        if (chunk_length == 0)
            chunk_length = 1;
    } else {
        chunk_length = std::numeric_limits<int32>::max();
    }

    int32 samp_offset = 0;
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;

    while (samp_offset < data.Dim()) {
        int32 samp_remaining = data.Dim() - samp_offset;
        int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

        kaldi::SubVector<kaldi::BaseFloat> wave_part(data, samp_offset, num_samp);
        _decode_wave(wave_part, delta_weights, samp_freq);

        samp_offset += num_samp;
    }
}

void Decoder::decode_raw_wav_audio(std::istream &wav_stream,
                                   const float &samp_freq,
                                   const int &data_bytes,
                                   const float &chunk_size) {
    kaldi::Matrix<kaldi::BaseFloat> wave_matrix;
    read_raw_wav_stream(wav_stream, data_bytes, wave_matrix);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> data(wave_matrix, 0);

    int32 chunk_length;
    if (chunk_size > 0) {
        chunk_length = int32(samp_freq * chunk_size);
        if (chunk_length == 0)
            chunk_length = 1;
    } else {
        chunk_length = std::numeric_limits<int32>::max();
    }

    int32 samp_offset = 0;
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;

    while (samp_offset < data.Dim()) {
        int32 samp_remaining = data.Dim() - samp_offset;
        int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

        kaldi::SubVector<kaldi::BaseFloat> wave_part(data, samp_offset, num_samp);
        _decode_wave(wave_part, delta_weights, samp_freq);

        samp_offset += num_samp;
    }
}

void Decoder::get_decoded_results(const int &n_best,
                                  utterance_results_t &results,
                                  const bool &word_level,
                                  const bool &bidi_streaming) {
    if (!bidi_streaming) {
        feature_pipeline_->InputFinished();
        decoder_->AdvanceDecoding();
        decoder_->FinalizeDecoding();
    }

    if (decoder_->NumFramesDecoded() == 0) {
        KALDI_WARN << "audio may be empty :: decoded no frames";
        return;
    }

    kaldi::CompactLattice clat;
    try {
        decoder_->GetLattice(true, &clat);
        find_alternatives(clat, n_best, results, word_level, model_, options);
    } catch (std::exception &e) {
        KALDI_ERR << "unexpected error during decoding lattice :: " << e.what(); 
    }
}

void Decoder::_decode_wave(kaldi::SubVector<kaldi::BaseFloat> &wave_part,
                           std::vector<std::pair<int32, kaldi::BaseFloat>> &delta_weights,
                           const kaldi::BaseFloat &samp_freq) {
    feature_pipeline_->AcceptWaveform(samp_freq, wave_part);

    if (silence_weighting_->Active() && feature_pipeline_->IvectorFeature() != NULL) {
        silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder());
        silence_weighting_->GetDeltaWeights(feature_pipeline_->NumFramesReady(),
                                            &delta_weights);
        feature_pipeline_->IvectorFeature()->UpdateFrameWeights(delta_weights);
    }

    decoder_->AdvanceDecoding();
}

} // namespace kaldiserve