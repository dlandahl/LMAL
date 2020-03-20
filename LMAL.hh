
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <iostream>
#include <fstream>

#define LMAL_For(n, k, N)    for (LMAL::uint n = k; n < N; n++)
#define LMAL_Sequence(N)     LMAL_For(n, 0, N) 
#define LMAL_Subscript(n, N) LMAL_For(n, 0, N) 
#define LMAL_Range(k, N)     LMAL_For(n, k, N) 

namespace LMAL {

	using uint = unsigned;
	
	inline float const tau = 6.28318530f;
	inline float const e = 2.71828182f;
	inline float const time_constant = -0.99967234f;
	inline float const butterworth_q = 1.f / std::sqrt(2);

	inline float sample_rate = 44100.f;
	inline float tempo = 120.000f;
	
	inline float
	seconds2samples(float seconds)
	{
		return sample_rate * seconds;
	}

	inline float
	milliseconds2samples(float milliseconds)
	{
		return sample_rate * milliseconds / 1000.f;
	}

	inline float
	samples2seconds(uint samples)
	{
		return float(samples / sample_rate);
	}

	inline float
	samples2milliseconds(uint samples)
	{
		return 1000.f * float(samples / sample_rate);
	}

	inline float
	linear2decibel(float linear)
	{
		return 20.f * std::log10(linear);
	}

	inline float
	decibel2linear(float decibels)
	{
		return std::pow(10, 0.05*decibels);
	}



	struct Buffer
	{
		uint size = 0;
		float* data = nullptr;
	};

	inline void
	allocate_buffer(Buffer* buffer, uint size)
	{
		buffer->data = (float*) malloc(sizeof( float ) * size);
		memset(buffer->data, 0, sizeof( float ) * size);
		buffer->size = size;
	}

	inline void
	deallocate_buffer(Buffer* buffer)
	{
		free(buffer->data);
		buffer->data = nullptr;
		buffer->size = 0;
	}

	inline void
	reallocate_buffer(Buffer* buffer, uint size)
	{
		free(buffer->data);
		allocate_buffer(buffer, size);
	}

	inline void
	operator+=(Buffer &buffer, float value)
	{
		LMAL_Sequence(buffer.size) buffer.data[n] += value;
	}

	inline void
	operator-=(Buffer &buffer, float value)
	{
		LMAL_Sequence(buffer.size) buffer.data[n] -= value;
	}

	inline void
	operator*=(Buffer &buffer, float value)
	{
		LMAL_Sequence(buffer.size) buffer.data[n] *= value;
	}

	inline void
	operator/=(Buffer &buffer, float value)
	{
		LMAL_Sequence(buffer.size) buffer.data[n] /= value;
	}



	struct Ringbuffer : Buffer
	{
		uint pointer = 0;

		float& operator[](int n) const {
			n += pointer;
			if (n < 0) n += size;
			return data[n % size];
		}

		Ringbuffer() = default;
		Ringbuffer(uint size);
		~Ringbuffer();
	};

	inline
	Ringbuffer::Ringbuffer(uint size)
	{
		allocate_buffer(this, size);
	}

	inline
	Ringbuffer::~Ringbuffer()
	{
		deallocate_buffer(this);
	}

	inline void
	clean_ringbuffer_pointer(Ringbuffer* ringbuffer)
	{
		while (ringbuffer->pointer > ringbuffer->size) ringbuffer->pointer -= ringbuffer->size;
		while (ringbuffer->pointer < 0) ringbuffer->pointer += ringbuffer->size;
	}

	inline void
	move_ringbuffer_pointer(Ringbuffer* ringbuffer, int steps)
	{
		ringbuffer->pointer += steps;
		clean_ringbuffer_pointer(ringbuffer);
	}

	inline void
	set_ringbuffer_pointer(Ringbuffer* ringbuffer, int position)
	{
		ringbuffer->pointer = position;
		clean_ringbuffer_pointer(ringbuffer);
	}

	inline void
	increment_ringbuffer_pointer(Ringbuffer* ringbuffer)
	{
		ringbuffer->pointer++;
		clean_ringbuffer_pointer(ringbuffer);
	}



	struct ZeroPaddedBuffer;
	struct BufferAccessProxy
	{
		ZeroPaddedBuffer* buffer;
		int index;
		bool out_of_bounds;

		BufferAccessProxy(ZeroPaddedBuffer* buffer, int index, bool out_of_bounds)
			: buffer(buffer), index(index), out_of_bounds(out_of_bounds) { }

		operator float();
		void operator=(float);
		void operator=(BufferAccessProxy);
	};

	struct ZeroPaddedBuffer : Buffer
	{
		bool allow_out_of_bounds_writing = false;
		
		BufferAccessProxy operator[](int index) {
			if (index < 0 || index >= (long int) size) return BufferAccessProxy(this, index, true);
			return BufferAccessProxy(this, index, false);
		}

		ZeroPaddedBuffer() = default;
		ZeroPaddedBuffer(uint);
		~ZeroPaddedBuffer();
	};

	inline
	ZeroPaddedBuffer::ZeroPaddedBuffer(uint size)
	{
		allocate_buffer(this, size);
	}

	inline
	ZeroPaddedBuffer::~ZeroPaddedBuffer()
	{
		deallocate_buffer(this);
	}

	inline
	BufferAccessProxy::operator float()
	{
		if (out_of_bounds) return 0.f;
		return buffer->data[index];
	}

	inline void
	BufferAccessProxy::operator=(float value)
	{
		if (out_of_bounds && buffer->allow_out_of_bounds_writing) return;
		assert(!out_of_bounds);
		buffer->data[index] = value;
	}
	
	inline void
	BufferAccessProxy::operator=(BufferAccessProxy proxy)
	{
		if (out_of_bounds && buffer->allow_out_of_bounds_writing) return;
		assert(!out_of_bounds);
		buffer->data[index] = (float) (*proxy.buffer)[proxy.index];
	}
	
	inline float
	read_value_interpolated(ZeroPaddedBuffer &buffer, float index)
	{
		float v1 = std::floor(index);
		float v2 = std::ceil (index);
		float ratio = index - v1;

		v1 = buffer[v1];
		v2 = buffer[v2];
	
		return (1-ratio) * v1 + ratio * v2;
	}
 
	inline float
	read_value_interpolated(Ringbuffer &buffer, float index)
	{
		float v1 = std::floor(index);
		float v2 = std::ceil (index);
		float ratio = index - v1;

		v1 = buffer[v1];
		v2 = buffer[v2];
	
		return (1-ratio) * v1 + ratio * v2;
	}



	struct Queue
	{
		Ringbuffer buffer;
		uint start_pointer = 0;
		uint length = 0;

		Queue(uint max_size) : buffer(max_size) {}
	};

	inline void
	push(Queue* queue, float value)
	{
		increment_ringbuffer_pointer(&queue->buffer);
		queue->buffer[0] = value;
		queue->length++;
		assert(queue->buffer.pointer != queue->start_pointer);
	}

	inline float 
	pop(Queue* queue)
	{
		assert(queue->buffer.pointer != queue->start_pointer);
		queue->start_pointer++;
		queue->length--;
		return queue->buffer[queue->start_pointer];
	}



	inline void
	normalise(Buffer* buffer)
	{
		float max = 0;
		LMAL_Sequence(buffer->size) if (buffer->data[n] > max) max = buffer->data[n];
		float value = 1.f / max;
		LMAL_Sequence(buffer->size) buffer->data[n] = buffer->data[n] * value;
	}

	inline float
	get_rms(float* data, uint N)
	{
		float value = 0;
		LMAL_Sequence(N) value += data[n] * data[n];
		value /= (float) N;
		return std::sqrt(value);
	}



	struct RawDiskData
	{
		char const * path;
	};

	inline void
	read_file_into_buffer(RawDiskData* input_file, Buffer* buffer)
	{
		std::fstream file(input_file->path, std::fstream::in | std::fstream::binary);
		LMAL_Sequence(buffer->size) {
			file.read((char *) &(buffer->data[n]), sizeof(float));
		}
	}

	inline void
	write_buffer_into_file(Buffer* buffer, RawDiskData* output_file)
	{
		std::fstream file(output_file->path, std::fstream::out | std::fstream::binary);
		LMAL_Sequence(buffer->size) {
			float value = buffer->data[n];
			file.write((char const*) &value, sizeof(float));
		}
	}



	inline void
	apply_hann_window(float* data, uint N)
	{
		LMAL_Sequence(N)
		{
			float val = std::sin((tau * n) / (2 * N));
			data[n] *= val * val;
		}
	}

	inline void
	apply_hann_window(Buffer* buffer)
	{
		apply_hann_window(buffer->data, buffer->size);
	}



	struct ComplexNumber
	{
		float real = 0.f;
		float imag = 0.f;
	};

	inline ComplexNumber
	operator*(ComplexNumber a, ComplexNumber b)
	{
		ComplexNumber out;
		out.real = a.real * b.real - a.imag * b.imag;
		out.imag = a.imag * b.real + a.real * b.imag;
		return out;
	}

	inline ComplexNumber
	operator+(ComplexNumber a, ComplexNumber b)
	{
		ComplexNumber out;
		out.real = a.real + b.real;
		out.imag = a.imag + b.imag;
		return out;
	}

	inline ComplexNumber
	operator-(ComplexNumber a, ComplexNumber b)
	{
		ComplexNumber out;
		out.real = a.real - b.real;
		out.imag = a.imag - b.imag;
		return out;
	}

	inline ComplexNumber
	complex_from_polar(float theta, float radius)
	{
		ComplexNumber out;
		out.real = std::cos(theta) * radius;
		out.imag = std::sin(theta) * radius;
		return out;
	}

	inline float
	complex_magnitude(ComplexNumber num)
	{
		return std::sqrt(num.real * num.real + num.imag * num.imag);
	}

	inline float
	complex_phase(ComplexNumber num)
	{
		return std::atan(num.imag / num.real);
	}
	
	inline void
	perform_fft(ComplexNumber* input_data, uint N)
	{
		if (N < 2) return;
		assert(!(N % 2));

		uint M = N / 2;
		
		auto temp = (ComplexNumber*) malloc(sizeof( ComplexNumber ) * M);
		LMAL_Sequence(M) temp[n] = input_data[n * 2 + 1];
		LMAL_Sequence(M) input_data[n] = input_data[n * 2];
		LMAL_Sequence(M) input_data[n + M] = temp[n];
		free(temp);

		perform_fft(input_data, M);
		perform_fft(input_data + M, M);

		LMAL_Subscript(k, M)
		{
			ComplexNumber even = input_data[k];
			ComplexNumber odd  = input_data[k + M];
		
			ComplexNumber w   = complex_from_polar(-tau * float(k) / N, 1.f) * odd;
			input_data[k]     = even + w;
			input_data[k + M] = even - w;
		}
	}

	inline ComplexNumber*
	allocate_and_perform_fft(float* input_data, uint N)
	{
		auto complex_data = (ComplexNumber*) malloc(sizeof( ComplexNumber ) * N);
		LMAL_Sequence(N)
			complex_data[n] = ComplexNumber { input_data[n], 0.f };

		perform_fft(complex_data, N);
		return complex_data;
	}



	struct SineOscillator
	{
		float frequency;
		float phase = 0.f;
		float phase_offset = 0.f;

		float operator()() {
			float value = std::sin(phase * tau + phase_offset);
			phase += frequency / sample_rate;
			if (phase >= 1.f) phase--;
			return value;
		}
	};

	struct SawOscillator
	{
		float frequency;
		float phase = 0.f;

		float operator()() {
			phase += 2.f * frequency / sample_rate;
			if (phase > 1.f) phase -= 2.f;
			return phase;
		}
	};

	struct SquareOscillator
	{
		float frequency;
		float phase = 0.f;
		float pulse_width = 0.5f;

		float operator()() {
			phase += frequency / sample_rate;
			if (phase >= 1.f) phase -= 1.f;
			if (phase < pulse_width) return 1.f;
			return -1.f;
		}
	};

	struct TriangleOscillator
	{
		float frequency;
		float phase = 0.f;

		float operator()() {
			phase += frequency / sample_rate;
			if (phase >= 1.f) phase -= 1.f;
			return 2.f * fabs(2.f * phase - 1.f) - 1.f;
		}
	};

	inline float
	noise()
	{
		return rand() / float (RAND_MAX / 2.f) - 1.f;
	}



	struct EnvelopeFollower
	{
		float _attack;
		float _release;
		
		float last_detector_value = 0.f;
	};

	struct Compressor
	{
		float threshold;
		float ratio;

		EnvelopeFollower follower;
	};

	inline void
	set_attack_release(EnvelopeFollower* follower, float attack_ms, float release_ms)
	{
		follower->_attack = std::exp(time_constant / (attack_ms * sample_rate * 0.001));
		follower->_release = std::exp(time_constant / (release_ms * sample_rate * 0.001));
	}

	inline float
	envelope_follow(EnvelopeFollower* follower, float sample)
	{
		auto& [attack, release, last_detector_value] = *follower;

		/*	
		float diff = std::abs(sample - detector_value);
		
		if (std::abs(sample) > detector_value)
			detector_value += diff / attack;
		
		if (std::abs(sample) < detector_value)
			detector_value -= diff / release;

		return detector_value;
		*/
		
		sample = std::fabs(sample);

		float detector_value = 0.f;
		if (sample > last_detector_value)
			detector_value = attack * (last_detector_value - sample) + sample;
		else
			detector_value = release * (last_detector_value - sample) + sample;

		if (detector_value < 0.f) detector_value = 0.f;

		last_detector_value = detector_value;
		return detector_value;
	}

	inline float
	compress(Compressor* compressor, float sample)
	{
		auto& [threshold, ratio, follower] = *compressor;
		float amplitude = envelope_follow(&follower, sample);
		float detector_value = linear2decibel(amplitude);
		float slope = 1.f - 1.f / ratio;
		float gain = slope * (threshold - detector_value);

		if (gain > 0) gain = 0;
	
		gain = decibel2linear(gain);

		return sample * gain;

		/*
		float detector_value = linear2decibel(envelope_follow(&follower, sample));
		float gain = 0.f;
		if (detector_value <= threshold) gain = threshold;
		else gain = threshold + (detector_value - threshold) / ratio;

		return decibel2linear(gain) * sample;
		*/
	}



	struct PhaseModulator
	{
		uint depth_in_samples;
		uint offset = 0;
		SineOscillator oscillator;
	};

	inline float
	calculate_phase_shift(PhaseModulator* modulator)
	{
		float phase_shift = modulator->oscillator();

		phase_shift = 0.5 + 0.5 * phase_shift;
		phase_shift *= modulator->depth_in_samples;
		phase_shift = -phase_shift;

		return phase_shift;
	}

	inline float
	phase_modulate(PhaseModulator* modulator, Ringbuffer &input)
	{
		auto& [depth_in_samples, offset,  oscillator] = *modulator;
		
		float phase_shift = calculate_phase_shift(modulator);
		float value = read_value_interpolated(input, offset + phase_shift);

		return value;
	}

	inline void
	phase_modulate(PhaseModulator* modulator, ZeroPaddedBuffer &input)
	{
		auto& [depth_in_samples, offset, oscillator] = *modulator;

		Buffer temp;
		allocate_buffer(&temp, input.size);
		
		LMAL_Sequence(input.size) {
			float phase_shift = calculate_phase_shift(modulator);
			temp.data[n] = read_value_interpolated(input, n + offset + phase_shift);
		}
		memcpy(input.data, temp.data, input.size * sizeof( float ));
		deallocate_buffer(&temp);
	}



	struct BiquadState
	{
		float xn = 0.f;
		float xn_1 = 0.f;
		float xn_2 = 0.f;
		
		float yn = 0.f;
		float yn_1 = 0.f;
		float yn_2 = 0.f;
	};

	struct BiquadraticFilter
	{
		BiquadState state;
		float frequency;
		float q = 1.f;
		float db_gain;

		double omega;
		double alpha;
		double cos_omega;
		double A;

		float b0, b1, b2;
		float a0, a1, a2;
	};

	using Biquad = BiquadraticFilter;

	inline void
	calculate_intermediates(Biquad* biquad)
	{
		biquad->omega = tau * biquad->frequency / sample_rate;
		biquad->alpha = std::sin(biquad->omega) / (2.f * biquad->q);
		biquad->cos_omega = std::cos(biquad->omega);
		biquad->A = std::pow(10, biquad->db_gain / 40.f);
	}

	inline void
	calculate_feedback_coefficients(Biquad* biquad)
	{
		biquad->a0 = 1 + biquad->alpha;
		biquad->a1 = -2 * biquad->cos_omega;
		biquad->a2 = 1 - biquad->alpha;
	}

	inline void
	calculate_lowpass_coefficients(Biquad* biquad)
	{
		biquad->b0 = (1 - biquad->cos_omega) / 2;
		biquad->b1 = 1 - biquad->cos_omega;
		biquad->b2 = biquad->b0;

		calculate_feedback_coefficients(biquad);
	}

	inline void
	calculate_highpass_coefficients(Biquad* biquad)
	{
		biquad->b0 = (1 + biquad->cos_omega) / 2;
		biquad->b1 = -(1 + biquad->cos_omega);
		biquad->b2 = biquad->b0;

		calculate_feedback_coefficients(biquad);
	}

	inline void
	calculate_bandpass_coefficients(Biquad* biquad)
	{
		biquad->b0 = biquad->q * biquad->alpha;
		biquad->b1 = 0;
		biquad->b2 = -biquad->q * biquad->alpha;

		calculate_feedback_coefficients(biquad);
	}

	inline void
	calculate_notch_coefficients(Biquad* biquad)
	{
		biquad->b0 = 1;
		biquad->b1 = -2 * biquad->cos_omega;
		biquad->b2 = 1;

		calculate_feedback_coefficients(biquad);
	}

	inline void
	calculate_allpass_coefficients(Biquad* biquad)
	{
		biquad->b0 = 1 - biquad->alpha;
		biquad->b1 = -2 * biquad->cos_omega;
		biquad->b2 = 1 + biquad->alpha;

		calculate_feedback_coefficients(biquad);
	}

	inline void
	calculate_peaking_coefficients(Biquad* biquad)
	{
		biquad->b0 = 1 + biquad->alpha * biquad->A;
		biquad->b1 = -2 * biquad->cos_omega;
		biquad->b2 = 1 - biquad->alpha * biquad->A;

		biquad->a0 = 1 + biquad->alpha / biquad->A;
		biquad->a1 = -2 * biquad->cos_omega;
		biquad->a2 = 1 - biquad->alpha / biquad->A;
	}

	inline void
	calculate_lowshelf_coefficients(Biquad* biquad)
	{
		double A = biquad->A;
		biquad->b0 =     A * ((A + 1) - (A - 1) * biquad->cos_omega + 2 * std::sqrt(A) * biquad->alpha);
		biquad->b1 = 2 * A * ((A - 1) - (A + 1) * biquad->cos_omega);
		biquad->b2 =     A * ((A + 1) - (A - 1) * biquad->cos_omega - 2 * std::sqrt(A) * biquad->alpha);

		biquad->a0 =      (A + 1) + (A - 1) * biquad->cos_omega + 2 * std::sqrt(A) * biquad->alpha;
		biquad->a1 = -2 * ((A - 1) + (A + 1) * biquad->cos_omega);
		biquad->a2 =      (A + 1) + (A - 1) * biquad->cos_omega - 2 * std::sqrt(A) * biquad->alpha;
	}

	inline void
	calculate_highshelf_coefficients(Biquad* biquad)
	{
		double A = biquad->A;
		biquad->b0 =      A * ((A + 1) + (A - 1) * biquad->cos_omega + 2 * std::sqrt(A) * biquad->alpha);
		biquad->b1 = -2 * A * ((A - 1) + (A + 1) * biquad->cos_omega);
		biquad->b2 =      A * ((A + 1) + (A - 1) * biquad->cos_omega - 2 * std::sqrt(A) * biquad->alpha);

		biquad->a0 =      (A + 1) - (A - 1) * biquad->cos_omega + 2 * std::sqrt(A) * biquad->alpha;
		biquad->a1 = 2 * ((A - 1) - (A + 1) * biquad->cos_omega);
		biquad->a2 =      (A + 1) - (A - 1) * biquad->cos_omega - 2 * std::sqrt(A) * biquad->alpha;
	}
	
	inline void
	apply_filtering(Biquad* biquad)
	{
		BiquadState& s = biquad->state;
		s.yn = (biquad->b0 * s.xn + biquad->b1 * s.xn_1 + biquad->b2 * s.xn_2	\
			- biquad->a1 * s.yn_1 - biquad->a2 * s.yn_2) / biquad->a0;
	}

	inline void
	apply_filtering(Biquad* biquad, Ringbuffer &x, Ringbuffer &y)
	{
		BiquadState& s = biquad->state;
		s.xn = x[0];
		s.xn_1 = x[-1];
		s.xn_2 = x[-2];

		s.yn_1 = y[-1];
		s.yn_2 = y[-2];
		apply_filtering(biquad);

		y[0] = s.yn;
	}

	inline void
	apply_filtering(Biquad* biquad, float &sample)
	{
		BiquadState& s = biquad->state;

		s.xn_2 = s.xn_1;
		s.xn_1 = s.xn;
		s.xn = sample;
		apply_filtering(biquad);
		
		s.yn_2 = s.yn_1;
		s.yn_1 = s.yn;
		sample = s.yn;
	}

	inline float
	get_filtered_value(Biquad* biquad, float sample)
	{
		apply_filtering(biquad, sample);
		return sample;
	}



	struct DelayLine
	{
		float feedback = 0.f;
		float time;

		float (*effect_loop)(float, void*) = nullptr;
		void* user_data;
		
		Ringbuffer delay_buffer;

		DelayLine(uint size) : delay_buffer(size) {}
	};

	inline float
	run_delay_line(DelayLine* ddl, float current_sample)
	{
		if (ddl->time < 0) return 0.f;
		increment_ringbuffer_pointer(&ddl->delay_buffer);
		float delayed_sample = read_value_interpolated(ddl->delay_buffer, -ddl->time);
		
		if (ddl->effect_loop) delayed_sample = ddl->effect_loop(delayed_sample, ddl->user_data);
		
		ddl->delay_buffer[0] = current_sample + delayed_sample * ddl->feedback;
		return delayed_sample;
	}



	struct AllpassFilter : DelayLine
	{
		AllpassFilter(uint order) : DelayLine(order) { time = order; };
	};

	inline float
	get_filtered_value(AllpassFilter* allpass_filter, float sample)
	{
		float delayed_sample = run_delay_line(allpass_filter, sample);
		return -allpass_filter->feedback * sample + allpass_filter->feedback * delayed_sample;
	}



	inline float
	quantize(float value, uint bits)
	{
		int64_t quantized = value * std::pow(2, bits);
		return float(quantized) / float(std::pow(2, bits));
	}



	inline int
	sign(float val)
	{
		return (val >= 0.f) - (val < 0.f);
	}

	using WaveshapingFunction = float (*)(float, float);

	inline WaveshapingFunction const
	TANH = [] (float sample, float k)
	{
		return std::tanh(sample * k) / std::tanh(k);
    };

	inline WaveshapingFunction const
	FEXP1 = [] (float sample, float k)
	{
		return float( sign(sample) * (1.f - std::pow(e, -std::fabs(k * sample))) / (1.f - std::pow(e, -k)) );
	};

	inline WaveshapingFunction const
	ARRY = [] (float sample, float k)
	{
		sample *= k;
		return (3.f * sample / 2.f) * (1 - (sample * sample / 3));
	};

	inline WaveshapingFunction const
	ATAN = [] (float sample, float k)
	{
		return atanf(sample * k) / atanf(k);
	};

	inline WaveshapingFunction const
	SQR = [] (float sample, float)
	{
		return sample * sample * sign(sample);
	};

	inline float
	apply_waveshaping(WaveshapingFunction function, float sample, float k = 1.f)
	{
		return function(sample, k);
	}

	inline float
	apply_asymetric_waveshaping(WaveshapingFunction function, float sample, float k, float l)
	{
		if (sample > 0.f) return function(sample, k);
		return function(sample, l);
	}
}
