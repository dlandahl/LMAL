
#include "LMAL.hh"

int main()
{
	LMAL::ZeroPaddedBuffer data(500000);
	LMAL::ZeroPaddedBuffer out(500000);
	LMAL::SineOscillator osc;
	osc.frequency = 700.f;
	//for (uint n = 0; n < data.size / 10; n++) data[n] = osc();

	LMAL::RawDiskData guitar_file;
	guitar_file.path = "Guitar.raw";
	LMAL::read_file_into_buffer(&guitar_file, &data);
	
	normalise(&data);
	//data[100] = 1.f;
	

	
	float reverb_time = 1.5f;
	uint const NUM_COMBS = 10;
	uint const NUM_APFS = 2;

	uint delay_size = LMAL::milliseconds2samples(64);
	uint apf_order = LMAL::milliseconds2samples(64);
	
	float comb_lengths[NUM_COMBS] = { 29.671, 32.375, 34.168, 36.834, 38.151, 39.466, 41.111, 42.783, 44.444, 45.197 };
	float apf_lengths[NUM_APFS] = { 8.775, 1.624 };

	LMAL::DelayLine combs[NUM_COMBS] { [0 ... NUM_COMBS-1] = delay_size };
	LMAL::AllpassFilter apfs[NUM_APFS] { [0 ... NUM_APFS-1] =  apf_order };

	for (uint n = 0; n < NUM_COMBS; n++) {
		combs[n].time = LMAL::milliseconds2samples(comb_lengths[n]);
		combs[n].feedback = std::pow(10.f, (-3.f * combs[n].time * (1.f / LMAL::sample_rate)) / reverb_time);
	}

	for (uint n = 0; n < NUM_APFS; n++) {
		apfs[n].time = LMAL::milliseconds2samples(apf_lengths[n]);
		apfs[n].feedback = -0.6;
	}

	LMAL::Biquad hpf;
	hpf.q = 0.1f;
	hpf.frequency = 500.f;
	calculate_intermediates(&hpf);
	calculate_highpass_coefficients(&hpf);

	LMAL::SineOscillator filter_mod1;
	LMAL::SineOscillator filter_mod2;
	filter_mod1.frequency = 0.05;
	filter_mod2.frequency = 0.07;
	
	for (uint s = 0; s < data.size; s++) {
		float output = 0.f;
		apfs[0].time = LMAL::milliseconds2samples(5.23f + filter_mod1());
		apfs[0].time = LMAL::milliseconds2samples(6.51f + filter_mod2());
		
		for (uint n = 0; n < NUM_COMBS / 2; n++) output += run_delay_line(&combs[n], data[s]);
		for (uint n = 0; n < NUM_APFS / 2; n++) output = get_filtered_value(&apfs[n], output);
		for (uint n = NUM_COMBS / 2; n < NUM_COMBS; n++) output += run_delay_line(&combs[n], data[s]);
		for (uint n = NUM_APFS / 2; n < NUM_APFS; n++) output = get_filtered_value(&apfs[n], output);
		output = get_filtered_value(&hpf, output);
		out[s] = output / 3.f ;//+ data[s + 2000];
	}



	normalise(&out);
	
	LMAL::RawDiskData file;
	file.path = "Reverb.raw";
	LMAL::write_buffer_into_file(&out, &file);
}
