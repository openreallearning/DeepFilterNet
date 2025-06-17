#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <sndfile.h>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <array>
#include "enhance.h"

struct Config {
    int sr = 0;
    int fft_size = 0;
    int hop_size = 0;
    int nb_erb = 0;
    int nb_df = 0;
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
            if (!std::getline(ls, val)) continue;
            auto trim = [](std::string s) {
                size_t b = s.find_first_not_of(" \t");
                size_t e = s.find_last_not_of(" \t");
                if (b == std::string::npos) return std::string();
                return s.substr(b, e - b + 1);
            };
            key = trim(key); val = trim(val);
            if (key == "sr") c.sr = std::stoi(val);
            else if (key == "fft_size") c.fft_size = std::stoi(val);
            else if (key == "hop_size") c.hop_size = std::stoi(val);
            else if (key == "nb_erb") c.nb_erb = std::stoi(val);
            else if (key == "nb_df") c.nb_df = std::stoi(val);
        }
    }
    return c;
}

static std::vector<float> hann(int n) {
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i)
        w[i] = 0.5f - 0.5f * std::cos(2*M_PI*i/n);
    return w;
}

static void dft(const std::vector<float> &in, std::vector<std::complex<float>> &out) {
    size_t N = in.size();
    out.assign(N, {});
    for (size_t k = 0; k < N; ++k) {
        std::complex<float> sum(0.f, 0.f);
        for (size_t n = 0; n < N; ++n) {
            float angle = -2*M_PI*k*n/N;
            sum += std::complex<float>(std::cos(angle), std::sin(angle)) * in[n];
        }
        out[k] = sum;
    }
}

static void idft(const std::vector<std::complex<float>> &in, std::vector<float> &out) {
    size_t N = in.size();
    out.assign(N, 0.f);
    for (size_t n = 0; n < N; ++n) {
        std::complex<float> sum(0.f,0.f);
        for (size_t k = 0; k < N; ++k) {
            float angle = 2*M_PI*k*n/N;
            sum += in[k] * std::complex<float>(std::cos(angle), std::sin(angle));
        }
        out[n] = sum.real()/N;
    }
}

int enhance_file(const std::string &model_tar,
                 const std::string &in_wav,
                 const std::string &out_wav) {
    char tmpdir[] = "/tmp/df_modelXXXXXX";
    if(!mkdtemp(tmpdir)) {
        std::cerr << "Failed to create temp directory" << std::endl;
        return 1;
    }
    std::string cmd = "tar -xzf " + model_tar + " -C " + tmpdir;
    if (std::system(cmd.c_str()) != 0) {
        std::cerr << "Failed to extract model" << std::endl;
        return 1;
    }
    std::filesystem::path base = std::filesystem::path(tmpdir)/"tmp"/"export";
    Config cfg = parse_config((base/"config.ini").string());
    if(cfg.fft_size==0){
        std::cerr << "Could not parse config" << std::endl; return 1;
    }

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "df");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    Ort::Session enc(env, (base/"enc.onnx").c_str(), opts);
    Ort::Session erb_dec(env, (base/"erb_dec.onnx").c_str(), opts);

    size_t n_freq = cfg.fft_size/2+1;
    auto window = hann(cfg.fft_size);

    SF_INFO info{};
    SNDFILE *infile = sf_open(in_wav.c_str(), SFM_READ, &info);
    if(!infile){ std::cerr << "Failed to open input" << std::endl; return 1; }
    if(info.channels!=1){ std::cerr << "Only mono" << std::endl; return 1; }

    std::vector<float> audio(info.frames);
    sf_readf_float(infile, audio.data(), info.frames);
    sf_close(infile);

    size_t hop = cfg.hop_size;
    size_t frames = (audio.size()+hop-1)/hop;
    std::vector<float> out(audio.size()+cfg.fft_size,0.f);
    std::vector<float> frame(cfg.fft_size);
    std::vector<std::complex<float>> spec(cfg.fft_size);
    std::vector<float> erb_feat(cfg.nb_erb);
    std::array<int64_t,4> enc_in1_shape{1,1,1,cfg.nb_erb};
    std::array<int64_t,4> enc_in2_shape{1,2,1,cfg.nb_df};
    std::vector<float> spec_feat(cfg.nb_df*2);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* enc_input_names[] = {"feat_erb","feat_spec"};
    const char* enc_output_names[] = {"e0","e1","e2","e3","emb"};
    const char* dec_input_names[] = {"emb","e3","e2","e1","e0"};
    const char* dec_output_names[] = {"m"};

    for(size_t f=0; f<frames; ++f){
        size_t start = f*hop;
        for(size_t i=0;i<cfg.fft_size;++i){
            float s = 0.f;
            if(start+i<audio.size()) s = audio[start+i];
            frame[i] = s * window[i];
        }
        dft(frame,spec);
        for(int b=0;b<cfg.nb_erb;++b){
            size_t b_start = b*n_freq/cfg.nb_erb;
            size_t b_end = (b+1)*n_freq/cfg.nb_erb;
            float acc=0.f; size_t count=0;
            for(size_t k=b_start;k<b_end;++k){
                acc += std::abs(spec[k]);
                ++count;
            }
            erb_feat[b]=count?acc/count:0.f;
        }
        for(int k=0;k<cfg.nb_df;++k){
            spec_feat[k] = spec[k].real();
            spec_feat[cfg.nb_df+k] = spec[k].imag();
        }
        std::vector<Ort::Value> enc_inputs;
        enc_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem_info, erb_feat.data(), cfg.nb_erb, enc_in1_shape.data(), enc_in1_shape.size()));
        enc_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem_info, spec_feat.data(), cfg.nb_df*2, enc_in2_shape.data(), enc_in2_shape.size()));
        auto enc_out = enc.Run(Ort::RunOptions{nullptr}, enc_input_names, enc_inputs.data(), 2, enc_output_names, 5);
        const char* dec_inputs_names[] = {"emb","e3","e2","e1","e0"};
        std::array<Ort::Value,5> dec_inputs{
            std::move(enc_out[4]),
            std::move(enc_out[3]),
            std::move(enc_out[2]),
            std::move(enc_out[1]),
            std::move(enc_out[0])
        };
        auto m_out = erb_dec.Run(Ort::RunOptions{nullptr}, dec_inputs_names, dec_inputs.data(), dec_inputs.size(), dec_output_names, 1);
        float* gains = m_out[0].GetTensorMutableData<float>();
        std::vector<float> gain_freq(n_freq,1.f);
        for(size_t k=0;k<n_freq;++k){
            size_t b = k*cfg.nb_erb/n_freq;
            gain_freq[k] = gains[b];
        }
        for(size_t k=0;k<n_freq;++k){
            spec[k] *= gain_freq[k];
        }
        std::vector<float> time(cfg.fft_size);
        idft(spec,time);
        for(size_t i=0;i<cfg.fft_size;++i){
            out[start+i] += time[i]*window[i];
        }
    }

    SF_INFO outinfo = info;
    SNDFILE *outfile = sf_open(out_wav.c_str(), SFM_WRITE, &outinfo);
    if(!outfile){ std::cerr << "Failed to open output" << std::endl; return 1; }
    sf_writef_float(outfile,out.data(),info.frames);
    sf_close(outfile);
    std::filesystem::remove_all(tmpdir);
    return 0;
}

#ifndef DF_NO_MAIN
int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.tar.gz> <input.wav> <output.wav>\n";
        return 1;
    }
    return enhance_file(argv[1], argv[2], argv[3]);
}
#endif

