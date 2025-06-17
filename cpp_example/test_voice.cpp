#include "enhance.h"
#include <sndfile.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <string>

static float snr(const std::vector<float>& ref, const std::vector<float>& target) {
    double ref_e = 0.0;
    double noise_e = 0.0;
    for(size_t i=0;i<ref.size();++i){
        double d = ref[i]-target[i];
        ref_e += ref[i]*ref[i];
        noise_e += d*d;
    }
    return 10.0f * std::log10(ref_e/noise_e);
}

static std::vector<float> read_wav(const std::string& path, int& sr) {
    SF_INFO info{};
    SNDFILE* f = sf_open(path.c_str(), SFM_READ, &info);
    if(!f){ std::cerr << "failed to open " << path << "\n"; exit(1); }
    sr = info.samplerate;
    std::vector<float> data(info.frames);
    sf_readf_float(f, data.data(), info.frames);
    sf_close(f);
    return data;
}

int main() {
    int sr=0;
    auto noisy = read_wav("../../assets/noisy_snr0.wav", sr);
    auto clean = read_wav("../../assets/clean_freesound_33711.wav", sr);
    clean.resize(sr); // use 1 second
    noisy.resize(sr);

    SF_INFO info{}; info.samplerate = sr; info.channels = 1; info.format = SF_FORMAT_WAV|SF_FORMAT_PCM_16;
    SNDFILE* f = sf_open("in.wav", SFM_WRITE, &info);
    sf_writef_float(f, noisy.data(), noisy.size());
    sf_close(f);

    if(enhance_file("../../models/DeepFilterNet3_onnx.tar.gz", "in.wav", "enhanced.wav") != 0) {
        std::cerr << "enhance failed\n"; return 1;
    }
    int sr2=0;
    auto enhanced = read_wav("enhanced.wav", sr2);
    enhanced.resize(sr);
    float snr_noisy = snr(clean, noisy);
    float snr_enh = snr(clean, enhanced);
    if(snr_enh > snr_noisy) return 0;
    std::cerr << "SNR did not improve\n";
    return 1;
}
