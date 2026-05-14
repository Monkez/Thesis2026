import numpy as np

def simulate_chunk_conv(chunk_size, N0_z, max_iter, scaling_factor,
                 num_channel_bits, info_len, m, n_rate, g_matrix, S, sym_matrix):
    # Re-initialize objects to avoid pickling overhead and issues
    from python_code.encoders import RSCEncoder
    from python_code.decoders import SISODecoder
    from python_code.demodulation import softdemod_qam, soft_demapper
    from python_code.channel import awgn_channel
    from python_code.modulation import modulate_bits
    from python_code.interleavers import random_interleaver
    
    encoder = RSCEncoder(g_matrix)
    decoder = SISODecoder(g_matrix, code_type=0, dec_type=0)
    
    bit_errors = np.zeros(max_iter)
    block_errors = 0
    
    for _ in range(chunk_size):
        u = np.random.randint(0, 2, info_len)
        v = encoder.encode(u)
        alpha = random_interleaver(num_channel_bits)
        vv = v[alpha]
        chan_in = modulate_bits(vv, S)
        chan_out = awgn_channel(chan_in, N0_z)
        demod = softdemod_qam(chan_out, N0_z, S)
        
        La = np.zeros(num_channel_bits)
        Le = La[alpha].copy()
        b_llr = np.zeros(num_channel_bits)
        SF = np.ones(num_channel_bits) * scaling_factor
        
        for iteration in range(max_iter):
            Le = (La - b_llr) * SF
            Le_pi = Le[alpha]
            b_llr[alpha] = soft_demapper(demod, Le_pi, sym_matrix)
            
            input_u = np.zeros(info_len)
            input_c = np.concatenate([b_llr, np.zeros(m * n_rate)])
            output_u, output_c = decoder.decode(input_u, input_c)
            La = output_c[:num_channel_bits]
            
            vhat = ((np.sign(La) + 1) / 2).astype(int)
            errors = np.sum(v != vhat)
            bit_errors[iteration] += errors
            
        if errors > 0:
            block_errors += 1
            
    return chunk_size, block_errors, bit_errors


def simulate_chunk_ham(chunk_size, N0_z, max_iter, scaling_factor,
                     num_channel_bits, frame_len, k_info, n_code, H, G, S, sym_matrix, alpha):
    from python_code.decoders import DualDecoder
    from python_code.demodulation import softdemod_qam, soft_demapper
    from python_code.channel import awgn_channel
    from python_code.modulation import modulate_bits
    
    dual_dec = DualDecoder(H)
    
    bit_errors = np.zeros(max_iter)
    block_errors = 0
    
    for _ in range(chunk_size):
        u = np.random.randint(0, 2, frame_len * k_info)
        u_reshaped = u.reshape((frame_len, k_info))
        v = (u_reshaped @ G) % 2
        v = v.flatten()

        vv = v[alpha]
        chan_in = modulate_bits(vv, S)
        chan_out = awgn_channel(chan_in, N0_z)
        demod = softdemod_qam(chan_out, N0_z, S)

        La = np.zeros(num_channel_bits)
        Le = La[alpha].copy()
        b_llr = np.zeros(num_channel_bits)
        SF = np.ones(num_channel_bits) * scaling_factor

        for iteration in range(max_iter):
            Le = (La - b_llr) * SF
            Le_pi = Le[alpha]

            b_llr_temp = soft_demapper(demod, Le_pi, sym_matrix)
            b_llr[alpha] = b_llr_temp

            La = dual_dec.decode_frame(b_llr)

            vhat = ((np.sign(La) + 1) / 2).astype(int)
            errors = np.sum(v != vhat)
            bit_errors[iteration] += errors

        if errors > 0:
            block_errors += 1

    return chunk_size, block_errors, bit_errors


def simulate_chunk_nonlinear_conv(chunk_size, N0_z, max_iter, scaling_factor,
                 num_channel_bits, info_len, m, n_rate, g_matrix, S, sym_matrix, 
                 cfo_norm, a_sat, p_rapp, alpha_pm):
    from python_code.encoders import RSCEncoder
    from python_code.decoders import SISODecoder
    from python_code.demodulation import softdemod_qam, soft_demapper
    from python_code.channel import cfopa_channel
    from python_code.modulation import modulate_bits
    from python_code.interleavers import random_interleaver
    import numpy as np
    
    encoder = RSCEncoder(g_matrix)
    decoder = SISODecoder(g_matrix, code_type=0, dec_type=0)
    
    bit_errors = np.zeros(max_iter)
    block_errors = 0
    
    for _ in range(chunk_size):
        u = np.random.randint(0, 2, info_len)
        v = encoder.encode(u)
        alpha = random_interleaver(num_channel_bits)
        vv = v[alpha]
        chan_in = modulate_bits(vv, S)
        
        chan_out = cfopa_channel(chan_in, N0_z, a_sat=a_sat, p_rapp=p_rapp, 
                                 alpha_pm=alpha_pm, cfo_norm=cfo_norm)
                                 
        demod = softdemod_qam(chan_out, N0_z, S)
        
        La = np.zeros(num_channel_bits)
        Le = La[alpha].copy()
        b_llr = np.zeros(num_channel_bits)
        SF = np.ones(num_channel_bits) * scaling_factor
        
        for iteration in range(max_iter):
            Le = (La - b_llr) * SF
            Le_pi = Le[alpha]
            b_llr[alpha] = soft_demapper(demod, Le_pi, sym_matrix)
            
            input_u = np.zeros(info_len)
            input_c = np.concatenate([b_llr, np.zeros(m * n_rate)])
            output_u, output_c = decoder.decode(input_u, input_c)
            La = output_c[:num_channel_bits]
            
            vhat = ((np.sign(La) + 1) / 2).astype(int)
            errors = np.sum(v != vhat)
            bit_errors[iteration] += errors
            
        if errors > 0:
            block_errors += 1
            
    return chunk_size, block_errors, bit_errors
