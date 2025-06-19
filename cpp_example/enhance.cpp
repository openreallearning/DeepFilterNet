#include "enhance.h"
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <sndfile.h>
#include <sstream>
#include <string>
#include <vector>

struct Config {
  int sr = 0;
  int fft_size = 0;
  int hop_size = 0;
  int nb_erb = 0;
  int nb_df = 0;
  int df_order = 0;
  int conv_lookahead = 0;
  int df_lookahead = 0;
};

static Config parse_config(const std::string &path) {
  Config c;
  std::ifstream f(path);
  std::string line;
  while (std::getline(f, line)) {
    std::istringstream ls(line);
    std::string key;
    if (std::getline(ls, key, '=')) {
      std::string val;
      if (!std::getline(ls, val))
        continue;
      auto trim = [](std::string s) {
        size_t b = s.find_first_not_of(" \t");
        size_t e = s.find_last_not_of(" \t");
        if (b == std::string::npos)
          return std::string();
        return s.substr(b, e - b + 1);
      };
      key = trim(key);
      val = trim(val);
      if (key == "sr")
        c.sr = std::stoi(val);
      else if (key == "fft_size")
        c.fft_size = std::stoi(val);
      else if (key == "hop_size")
        c.hop_size = std::stoi(val);
      else if (key == "nb_erb")
        c.nb_erb = std::stoi(val);
      else if (key == "nb_df")
        c.nb_df = std::stoi(val);
      else if (key == "df_order")
        c.df_order = std::stoi(val);
      else if (key == "conv_lookahead")
        c.conv_lookahead = std::stoi(val);
      else if (key == "df_lookahead")
        c.df_lookahead = std::stoi(val);
    }
  }
  return c;
}

static std::vector<float> hann(int n) {
  std::vector<float> w(n);
  for (int i = 0; i < n; ++i)
    w[i] = 0.5f - 0.5f * std::cos(2 * M_PI * i / n);
  return w;
}

static void dft(const std::vector<float> &in,
                std::vector<std::complex<float>> &out) {
  size_t N = in.size();
  out.assign(N, {});
  for (size_t k = 0; k < N; ++k) {
    std::complex<float> sum(0.f, 0.f);
    for (size_t n = 0; n < N; ++n) {
      float angle = -2 * M_PI * k * n / N;
      sum += std::complex<float>(std::cos(angle), std::sin(angle)) * in[n];
    }
    out[k] = sum;
  }
}

static void idft(const std::vector<std::complex<float>> &in,
                 std::vector<float> &out) {
  size_t N = in.size();
  out.assign(N, 0.f);
  for (size_t n = 0; n < N; ++n) {
    std::complex<float> sum(0.f, 0.f);
    for (size_t k = 0; k < N; ++k) {
      float angle = 2 * M_PI * k * n / N;
      sum += in[k] * std::complex<float>(std::cos(angle), std::sin(angle));
    }
    out[n] = sum.real() / N;
  }
}

int enhance_file(const std::string &model_tar, const std::string &in_wav,
                 const std::string &out_wav) {
  char tmpdir[] = "/tmp/df_modelXXXXXXX";
  if (!mkdtemp(tmpdir)) {
    std::cerr << "Failed to create temp directory" << std::endl;
    return 1;
  }
  std::string cmd = "tar -xzf " + model_tar + " -C " + tmpdir;
  if (std::system(cmd.c_str()) != 0) {
    std::cerr << "Failed to extract model" << std::endl;
    return 1;
  }
  std::filesystem::path base = std::filesystem::path(tmpdir) / "tmp" / "export";
  Config cfg = parse_config((base / "config.ini").string());
  if (cfg.fft_size == 0) {
    std::cerr << "Could not parse config" << std::endl;
    return 1;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "df");
  Ort::SessionOptions opts;
  opts.SetIntraOpNumThreads(1);
  Ort::Session enc(env, (base / "enc.onnx").c_str(), opts);
  Ort::Session erb_dec(env, (base / "erb_dec.onnx").c_str(), opts);
  Ort::Session df_dec(env, (base / "df_dec.onnx").c_str(), opts);

  size_t n_freq = cfg.fft_size / 2 + 1;
  auto window = hann(cfg.fft_size);

  SF_INFO info{};
  SNDFILE *infile = sf_open(in_wav.c_str(), SFM_READ, &info);
  if (!infile) {
    std::cerr << "Failed to open input" << std::endl;
    return 1;
  }
  if (info.channels != 1) {
    std::cerr << "Only mono" << std::endl;
    return 1;
  }

  std::vector<float> audio(info.frames);
  sf_readf_float(infile, audio.data(), info.frames);
  sf_close(infile);

  size_t hop = cfg.hop_size;
  size_t frames = (audio.size() + hop - 1) / hop;
  size_t lookahead = std::max(cfg.conv_lookahead, cfg.df_lookahead);
  // Process additional frames to flush lookahead at the end
  size_t proc_frames = frames + lookahead + cfg.conv_lookahead;
  size_t delay = (cfg.fft_size - hop) + lookahead * hop;
  std::vector<float> out(audio.size() + delay + cfg.fft_size, 0.f);
  std::vector<float> frame(cfg.fft_size);
  std::vector<std::complex<float>> spec(cfg.fft_size);
  // Store the noisy spectrum for each frame
  std::vector<std::vector<std::complex<float>>> spec_noisy(
      proc_frames, std::vector<std::complex<float>>(cfg.fft_size));
  for (size_t f = 0; f < proc_frames; ++f) {
    size_t start = f * hop;
    for (size_t i = 0; i < cfg.fft_size; ++i) {
      float s = 0.f;
      if (start + i < audio.size())
        s = audio[start + i];
      frame[i] = s * window[i];
    }
    dft(frame, spec);
    for (size_t k = 0; k < cfg.fft_size; ++k)
      spec_noisy[f][k] = spec[k];
  }

  // Processing buffer starts with the noisy spectrum
  auto spec_proc = spec_noisy;

  std::vector<float> erb_feat(cfg.nb_erb);
  std::array<int64_t, 4> enc_in1_shape{1, 1, 1, cfg.nb_erb};
  std::array<int64_t, 4> enc_in2_shape{1, 2, 1, cfg.nb_df};
  std::vector<float> spec_feat(cfg.nb_df * 2);

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char *enc_input_names[] = {"feat_erb", "feat_spec"};
  const char *enc_output_names[] = {"e0",  "e1",   "e2", "e3",
                                    "emb", "lsnr", "c0"};
  const char *dec_input_names[] = {"emb", "e3", "e2", "e1", "e0"};
  const char *dec_output_names[] = {"m"};

  for (size_t t = 0; t < proc_frames - cfg.conv_lookahead; ++t) {
    const auto &spec_in = spec_noisy[t + cfg.conv_lookahead];
    for (int b = 0; b < cfg.nb_erb; ++b) {
      size_t b_start = b * n_freq / cfg.nb_erb;
      size_t b_end = (b + 1) * n_freq / cfg.nb_erb;
      float acc = 0.f;
      size_t count = 0;
      for (size_t k = b_start; k < b_end; ++k) {
        acc += std::abs(spec_in[k]);
        ++count;
      }
      erb_feat[b] = count ? acc / count : 0.f;
    }
    for (int k = 0; k < cfg.nb_df; ++k) {
      spec_feat[k] = spec_in[k].real();
      spec_feat[cfg.nb_df + k] = spec_in[k].imag();
    }
    std::vector<Ort::Value> enc_inputs;
    enc_inputs.emplace_back(Ort::Value::CreateTensor<float>(
        mem_info, erb_feat.data(), cfg.nb_erb, enc_in1_shape.data(),
        enc_in1_shape.size()));
    enc_inputs.emplace_back(Ort::Value::CreateTensor<float>(
        mem_info, spec_feat.data(), cfg.nb_df * 2, enc_in2_shape.data(),
        enc_in2_shape.size()));
    auto enc_out = enc.Run(Ort::RunOptions{nullptr}, enc_input_names,
                           enc_inputs.data(), 2, enc_output_names, 7);

    auto c0 = std::move(enc_out[6]);
    auto emb = std::move(enc_out[4]);
    auto lsnr = enc_out[5].GetTensorMutableData<float>()[0];


    const float attention_limit_parameter = 100, // db
        min_db_threshould = -6.5,                 // db
        max_db_erb_threshould = 30,              // db
        max_db_df_threshould = 20;               // db

    // Only noise detected, do not apply gain
    bool only_noise_detected = lsnr < min_db_threshould;
    bool clean_speech_signal = lsnr > max_db_erb_threshould;
    bool only_little_noise_detected = lsnr > max_db_df_threshould;

    std::cout << "Local SNR: " << lsnr << " ";
    if (clean_speech_signal) std::cout << " clean ";
    if (only_noise_detected) std::cout << " NOISE ";
    if (only_little_noise_detected) std::cout << " noise ";
    std::cout << std::endl;

    std::vector<float> gain_freq(n_freq, 1);
    std::vector<std::complex<float>> spec_out = spec_proc[t];
    std::vector<std::complex<float>> spec_df(cfg.nb_df);

    if (!only_noise_detected && !clean_speech_signal) {
      // Apply ERB gains
      std::array<Ort::Value, 5> dec_inputs{
          std::move(emb), std::move(enc_out[3]), std::move(enc_out[2]),
          std::move(enc_out[1]), std::move(enc_out[0])};

      // ERB dec convolution
      auto m_out = erb_dec.Run(Ort::RunOptions{nullptr}, dec_input_names,
                               dec_inputs.data(), dec_inputs.size(),
                               dec_output_names, 1);
      emb = std::move(dec_inputs[0]);
      float *gains = m_out[0].GetTensorMutableData<float>();

      // Convert ERB gains to frequency gains
      for (size_t k = 0; k < n_freq; ++k) {
        // Find the ERB bin based on frequency
        size_t b = k * cfg.nb_erb / n_freq;
        gain_freq[k] = gains[b];
      }
      for (size_t k = 0; k < n_freq; ++k) {
        spec_out[k] *= gain_freq[k];
      }
      spec_proc[t] = spec_out; // store stage 1 output for DF history
    }

    if (!only_noise_detected && !clean_speech_signal && !only_little_noise_detected) {
      // Apply DF
      // DF dec convolution
      const char *df_dec_input_names[] = {"emb", "c0"};
      std::array<Ort::Value, 2> df_dec_inputs{std::move(emb), std::move(c0)};
      const char *df_dec_output_names[] = {"coefs"};
      auto df_out = df_dec.Run(Ort::RunOptions{nullptr}, df_dec_input_names,
                               df_dec_inputs.data(), df_dec_inputs.size(),
                               df_dec_output_names, 1);
      float *coefs = df_out[0].GetTensorMutableData<float>();
      std::fill(spec_df.begin(), spec_df.end(), std::complex<float>{0.f, 0.f});
      for (size_t o = 0; o < cfg.df_order; ++o) {
        int hist_idx = static_cast<int>(t) - static_cast<int>(cfg.df_order) + 1 +
                        static_cast<int>(o) + static_cast<int>(cfg.df_lookahead);
        if (hist_idx < 0 || hist_idx >= static_cast<int>(spec_proc.size()))
          continue;
        const auto &hist = spec_proc[hist_idx];
        for (size_t k = 0; k < cfg.nb_df; ++k) {
          size_t idx = k * (cfg.df_order * 2) + 2 * o;
          std::complex<float> c(coefs[idx], coefs[idx + 1]);
          spec_df[k] += hist[k] * c;
        }
      }
      for (size_t k = 0; k < cfg.nb_df; ++k)
        spec_out[k] *= spec_df[k];
    }

    if (false) {
      float atten_lim = std::pow(10.f, -attention_limit_parameter / 20.f);
      for (size_t k = 0; k < n_freq; ++k) {
        spec_out[k] = spec_out[k] * (1.f - atten_lim) + spec_out[k] * atten_lim;
      }
    }

    // Ensure conjugate symmetry for real iDFT
    for (size_t k = 1; k < n_freq - 1; ++k) {
      spec_out[cfg.fft_size - k] = std::conj(spec_out[k]);
    }

    std::vector<float> time(cfg.fft_size);
    idft(spec_out, time);
    size_t start = t * hop;
    for (size_t i = 0; i < cfg.fft_size; ++i) {
      if (only_noise_detected) out[start + i] = 0;
      else
        out[start + i] += time[i] * window[i];
    }
  }

  SF_INFO outinfo = info;
  SNDFILE *outfile = sf_open(out_wav.c_str(), SFM_WRITE, &outinfo);
  if (!outfile) {
    std::cerr << "Failed to open output" << std::endl;
    return 1;
  }
  sf_writef_float(outfile, out.data() + delay, info.frames);
  sf_close(outfile);
  std::filesystem::remove_all(tmpdir);
  return 0;
}

#ifndef DF_NO_MAIN
int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model.tar.gz> <input.wav> <output.wav>\n";
    return 1;
  }
  return enhance_file(argv[1], argv[2], argv[3]);
}
#endif
