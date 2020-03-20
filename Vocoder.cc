
#include "LMAL.hh"

int main()
{
	using namespace LMAL;
	
	sample_rate = 44100.f;
	int const size = 179569;
	
	RawDiskData voice_file;
	ZeroPaddedBuffer voice_data(size);
	voice_file.path = "Voice.raw";
	read_file_into_buffer(&voice_file, &voice_data);

	int const band_count = 4;

	Biquad modulator_bands[band_count];
	Biquad carrier_bands[band_count];
	Ringbuffer modulator_buffers[band_count];
	Ringbuffer carrier_buffers[band_count];
	
	EnvelopeFollower followers[band_count];

	for (int n = 0, f = 200; n < band_count; n++, f *= 2.3)
	{
		modulator_bands[n].frequency = f;
		carrier_bands[n].frequency = f;

		modulator_bands[n].q = 8;
		carrier_bands[n].q = 8;
		
		calculate_intermediates(&modulator_bands[n]);
		calculate_intermediates(&carrier_bands[n]);

		calculate_bandpass_coefficients(&modulator_bands[n]);
		calculate_bandpass_coefficients(&carrier_bands[n]);
		
		allocate_buffer(&modulator_buffers[n], 64);
		allocate_buffer(&carrier_buffers[n], 64);

		set_attack_release(&followers[n], 20, 20);
	}

	SawOscillator chord[4];
	chord[0].frequency = 110;
	chord[1].frequency = 440;
	chord[2].frequency = 262 * 2;
	chord[3].frequency = 330;
	
	Ringbuffer chord_buffer(2048);
	PhaseModulator flanger;
	flanger.depth_in_samples = 4000;
	flanger.oscillator.frequency = 0.1;
	flanger.offset = 1000;
	
	ZeroPaddedBuffer output_data(size);

	for (int n = 0; n < size; n++)
	{
		float saw_value = ( chord[0]() + chord[1]() + chord[2]() + chord[3]() ) / 4.f;
		chord_buffer[0] = saw_value;

		//saw_value = phase_modulate(&flanger, chord_buffer) + saw_value;
		saw_value /= 2.f;
		chord_buffer.pointer++;
		
		for (int b = 0; b < band_count; b++)
		{
			modulator_buffers[b][0] = voice_data[n];
			modulator_buffers[b][0] = get_filtered_value(&modulator_bands[b], modulator_buffers[b][0]);

		    envelope_follow(&followers[b], modulator_buffers[b][0]);

			carrier_buffers[b][0] = saw_value;
			carrier_buffers[b][0] = get_filtered_value(&carrier_bands[b], carrier_buffers[b][0]);

			carrier_buffers[b][0] *= followers[b].last_detector_value;

			output_data[n] = output_data[n] + carrier_buffers[b][0];
			
			modulator_buffers[b].pointer++;
			carrier_buffers[b].pointer++;
		}
	}

	RawDiskData output_file;
	output_file.path = "Vocoder.raw";
	normalise(&output_data);
	write_buffer_into_file(&output_data, &output_file);
}

