# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:14:25 2021

@author: Hatlab_3
"""
# import easygui
from plottr.apps.autoplot import autoplotDDH5, script, main
from plottr.data.datadict_storage import all_datadicts_from_hdf5
import matplotlib.pyplot as plt
import numpy as np
from data_processing.Helper_Functions import get_name_from_path, shift_array_relative_to_middle, log_normalize_to_row, select_closest_to_target
from data_processing.fitting.QFit import fit, plotRes, reflectionFunc
import inspect
from plottr.data import datadict_storage as dds, datadict as dd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.fftpack import dct, idct

# FS_filepath = r'Z:/Data/SA_2X_B1/fluxsweep/fits/2021-07-22/2021-07-22_0024_SA_2X_B1/2021-07-22_0024_SA_2X_B1.ddh5'
# Duff_filepath = r'Z:/Data/SA_2X_B1/duffing/2021-07-23/2021-07-23_0004_SA_2X_B1_duffing/2021-07-23_0004_SA_2X_B1_duffing.ddh5'
# save_filepath = r'Z:\Data\SA_2X_B1\duffing\fits'
#create a custom exception that handles fit errors in a more transparent way 

#%%Create measurement-based saver for the fit data. 

class fit_Duff_Measurement():
    '''
    outline: 
        - Take in an existing Wolfie-format duffing file and manual fit popt
        - run semiauto_fit for each generator power, return an array of popts in an N_currents x M_gen_powers array
        - use plotRes to debug fitter
        - generate duffing graph
    '''
    
    def __init__(self, name):
        #setup files
        self.name = name
        
    def create_file(self, save_filepath): 
        self.datadict = dd.DataDict(
            current = dict(unit='A'),
            gen_power = dict(unit = 'dBm'),
            
            base_resonant_frequency = dict(axes = ['current']),
            low_power_res_frequency = dict(axes = ['current']),
            
            driven_resonant_frequency= dict(axes = ['current', 'gen_power']), 
            driven_Qint = dict(axes = ['current', 'gen_power']), 
            driven_Qext = dict(axes = ['current', 'gen_power']),
            
            driven_resonant_frequency_error= dict(axes = ['current', 'gen_power']), 
            driven_Qint_error = dict(axes = ['current', 'gen_power']), 
            driven_Qext_error = dict(axes = ['current', 'gen_power']),
            
            res_shift_ref_undriven = dict(axes = ['current', 'gen_power']), 
            res_shift_ref_low = dict(axes = ['current', 'gen_power'])
        )
        self.datadir = save_filepath
        self.writer = dds.DDH5Writer( self.datadict,self.datadir, name=self.name)
        self.writer.__enter__()
        return None
    
    def load_data(self, Duff_filepath, FS_filepath, current_filt = None, 
                  current_name = 'current',  
                  gen_power_name = 'gen_power',
                  vna_freq_name = 'vna_frequency',
                  vna_pow_name = 'vna_power',
                  vna_phase_name = 'driven_vna_phase',
                  phase_unit = 'rad',
                  elec_delay=None,  # float or None
                  delay_unit='s',  # 's', 'ns', 'ps'
                  delay_sign=+1,  # +1 means phi += omega*tau; use -1 if your convention is opposite
                  delay_reference='zero'):

        # Duffing Data Extraction
        duff_dicts = all_datadicts_from_hdf5(Duff_filepath)
        duffDict = duff_dicts['data']
        uvphDict = duffDict.extract(vna_phase_name)
        uvpoDict = duffDict.extract(vna_pow_name)
        dvphDict = duffDict.extract(vna_phase_name)
        dvpoDict = duffDict.extract(vna_pow_name)

        if current_filt is None:
            lower = np.min(uvphDict.data_vals(current_name))
            upper = np.max(uvphDict.data_vals(current_name))
        else:
            lower, upper = current_filt

        filt = (uvphDict.data_vals(current_name) < upper) * (uvphDict.data_vals(current_name) > lower)

        self.undriven_vna_power = uvpoDict.data_vals(vna_pow_name)[filt]

        # frequency (store omega in rad/s as you already do)
        omega = uvphDict.data_vals(vna_freq_name)[filt] * 2 * np.pi
        self.vna_freqs = omega
        self.currents = uvphDict.data_vals(current_name)[filt]
        self.gen_powers = dvpoDict.data_vals(gen_power_name)[filt]
        self.driven_vna_power = dvpoDict.data_vals(vna_pow_name)[filt]

        # --- phase read + unit conversion ---
        if phase_unit.lower() == 'rad':
            uv_phase = uvphDict.data_vals(vna_phase_name)[filt]
            dv_phase = dvphDict.data_vals(vna_phase_name)[filt]
        elif phase_unit.lower() == 'deg':
            uv_phase = uvphDict.data_vals(vna_phase_name)[filt] / 360 * 2 * np.pi
            dv_phase = dvphDict.data_vals(vna_phase_name)[filt] / 360 * 2 * np.pi
        else:
            raise Exception("enter either 'deg' or 'rad'")

        # --- electrical delay correction (optional) ---
        if elec_delay is not None:
            # convert delay to seconds
            if delay_unit == 's':
                tau = float(elec_delay)
            elif delay_unit == 'ns':
                tau = float(elec_delay) * 1e-9
            elif delay_unit == 'ps':
                tau = float(elec_delay) * 1e-12
            else:
                raise Exception("delay_unit must be 's', 'ns', or 'ps'")

            # choose reference omega to reduce huge phase ramps (optional)
            if delay_reference == 'zero':
                omega_ref = 0.0
            elif delay_reference == 'mean':
                omega_ref = float(np.mean(omega))
            else:
                # allow user to pass a reference frequency in Hz
                # e.g. delay_reference=6e9
                omega_ref = float(delay_reference) * 2 * np.pi

            phase_ramp = delay_sign * (omega - omega_ref) * tau

            uv_phase = uv_phase + phase_ramp
            dv_phase = dv_phase + phase_ramp

        # assign to object
        self.undriven_vna_phase = uv_phase
        self.driven_vna_phase = dv_phase

        self.res_func, self.qint_func, self.qext_func = self.read_fs_data(FS_filepath)
        return None

    def explore_delay(self,
                      delay_values,
                      delay_unit='ns',
                      delay_sign=+1,
                      delay_reference='mean',
                      current_idx=0,
                      gen_power_idx=0,
                      figsize=(14, 10)):
        """
        Plot the effect of different electrical delay corrections on the phase trace
        of a single (current, gen_power) slice, so you can visually pick the best tau
        before running semiauto_fit.

        Parameters
        ----------
        delay_values : float or list/array of floats
            Delay values to try (in units given by delay_unit).
            Pass a single float to compare just "no correction" vs that one value.
        delay_unit : str
            'ns', 'ps', or 's'
        delay_sign : int
            +1  →  phi += sign * omega * tau   (same convention as load_data)
        delay_reference : str or float
            'zero', 'mean', or a frequency in Hz to subtract before applying the ramp.
        current_idx : int
            Index into np.unique(self.currents) to select which current slice to inspect.
        gen_power_idx : int
            Index into np.unique(self.gen_powers) to select which gen_power slice.
        figsize : tuple
            Figure size.

        Returns
        -------
        None  (just shows the plot)
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # --- unit conversion factor ---
        unit_map = {'s': 1.0, 'ns': 1e-9, 'ps': 1e-12}
        if delay_unit not in unit_map:
            raise ValueError("delay_unit must be 's', 'ns', or 'ps'")
        scale = unit_map[delay_unit]

        # --- normalise delay_values to a list ---
        if np.ndim(delay_values) == 0:
            delay_values = [float(delay_values)]
        else:
            delay_values = list(delay_values)

        # --- select the slice ---
        unique_currents = np.sort(np.unique(self.currents))
        unique_genpowers = np.sort(np.unique(self.gen_powers))

        if current_idx >= len(unique_currents):
            raise IndexError(f"current_idx={current_idx} out of range (have {len(unique_currents)} currents)")
        if gen_power_idx >= len(unique_genpowers):
            raise IndexError(f"gen_power_idx={gen_power_idx} out of range (have {len(unique_genpowers)} gen_powers)")

        c_sel = unique_currents[current_idx]
        p_sel = unique_genpowers[gen_power_idx]
        filt = (self.currents == c_sel) & (self.gen_powers == p_sel)

        omega = self.vna_freqs[filt]  # already in rad/s from load_data
        phase_raw = self.driven_vna_phase[filt]
        pow_trace = self.driven_vna_power[filt]
        freq_hz = omega / (2 * np.pi)

        # --- reference omega for ramp ---
        if delay_reference == 'zero':
            omega_ref = 0.0
        elif delay_reference == 'mean':
            omega_ref = float(np.mean(omega))
        else:
            omega_ref = float(delay_reference) * 2 * np.pi

        # --- build figure ---
        n_delays = len(delay_values)
        n_cols = min(3, n_delays + 1)  # +1 for the "no correction" panel
        n_rows = int(np.ceil((n_delays + 1) / n_cols))

        fig = plt.figure(figsize=figsize)
        fig.suptitle(
            f"Electrical Delay Explorer\n"
            f"Current = {c_sel * 1e3:.3f} mA,  Gen Power = {p_sel} dBm\n"
            f"(delay_sign={delay_sign:+d}, ref={delay_reference})",
            fontsize=12, fontweight='bold'
        )
        gs = gridspec.GridSpec(n_rows * 2, n_cols, figure=fig,
                               hspace=0.55, wspace=0.35)

        def _plot_panel(col_idx, row_idx, tau_s, label):
            ax_ph = fig.add_subplot(gs[row_idx * 2, col_idx])
            ax_iq = fig.add_subplot(gs[row_idx * 2 + 1, col_idx])

            ramp = delay_sign * (omega - omega_ref) * tau_s
            phase_corr = phase_raw + ramp
            lin = 10 ** (pow_trace / 20)
            real = lin * np.cos(phase_corr)
            imag_ = lin * np.sin(phase_corr)

            # Phase panel
            ax_ph.plot(freq_hz / 1e9, np.unwrap(phase_corr), color='steelblue', lw=1.2)
            ax_ph.set_title(label, fontsize=9, pad=3)
            ax_ph.set_xlabel('Freq (GHz)', fontsize=7)
            ax_ph.set_ylabel('Phase (rad)', fontsize=7)
            ax_ph.tick_params(labelsize=7)

            # IQ panel
            ax_iq.plot(real, imag_, color='darkorange', lw=1.0)
            ax_iq.set_aspect('equal', 'datalim')
            ax_iq.set_xlabel('Re', fontsize=7)
            ax_iq.set_ylabel('Im', fontsize=7)
            ax_iq.tick_params(labelsize=7)
            ax_iq.set_title('IQ loop', fontsize=8, pad=2)

        # No-correction panel
        _plot_panel(0, 0, 0.0, 'No correction (τ = 0)')

        # One panel per delay value
        for k, dv in enumerate(delay_values):
            tau_s = dv * scale
            panel = k + 1
            col = panel % n_cols
            row = panel // n_cols
            label = f'τ = {dv} {delay_unit}'
            _plot_panel(col, row, tau_s, label)

        plt.show()
        print(f"\nSlice info: current={c_sel} A, gen_power={p_sel} dBm, {np.sum(filt)} points")
        print(f"Freq range: {freq_hz.min() / 1e9:.4f} – {freq_hz.max() / 1e9:.4f} GHz")
        print(f"To apply the chosen delay, pass  elec_delay=<value>, delay_unit='{delay_unit}'  to load_data().")
    def read_fs_data(self, fs_filepath, interpolation = 'linear'):
        ret = all_datadicts_from_hdf5(fs_filepath)
        res_freqs = ret['data'].extract('base_resonant_frequency').data_vals('base_resonant_frequency')
        currents = ret['data'].extract('base_resonant_frequency').data_vals('current')
        Qexts = ret['data'].extract('base_Qext').data_vals('base_Qext')
        Qints = ret['data'].extract('base_Qint').data_vals('base_Qint')
        fs_res_fit_func = interp1d(currents, res_freqs, interpolation)
        fs_Qint_fit_func = interp1d(currents, Qints, interpolation)
        fs_Qext_fit_func = interp1d(currents, Qexts, interpolation)
        return fs_res_fit_func, fs_Qint_fit_func, fs_Qext_fit_func
    
    def save_fit(self, currents, gen_power,driven_popts, driven_pconvs, low_power_res_fit_func):
        print('saving!')
        for i, current in enumerate(np.unique(currents)): 
            driven_popt = driven_popts[i]
            driven_pconv = driven_pconvs[i]
            res_freq_ref = float(self.res_func(current))    
            res_freq_low_power = float(low_power_res_fit_func(current))
            self.writer.add_data(
                current = current, 
                gen_power = gen_power,
                
                base_resonant_frequency = res_freq_ref, 
                low_power_res_frequency = res_freq_low_power,
                
                driven_resonant_frequency = driven_popt[2]/(2*np.pi), 
                driven_Qint = driven_popt[1],
                driven_Qext = driven_popt[0],
                
                driven_resonant_frequency_error = np.sqrt(driven_pconv[2,2])/(2*np.pi),
                driven_Qint_error = np.sqrt(driven_pconv[1,1]),
                driven_Qext_error = np.sqrt(driven_pconv[0,0]),
                
                res_shift_ref_undriven = (driven_popt[2]/(2*np.pi)-res_freq_ref),
                res_shift_ref_low = (driven_popt[2]/(2*np.pi)-res_freq_low_power)
                
                )

        print('saved')

    def save_fit_batched(self, all_currents, all_gen_powers, all_popts, all_pconvs, low_power_res_fit_func):
        popts_arr = np.array(all_popts)
        pconvs_arr = np.array(all_pconvs)

        res_freq_refs = np.array([float(self.res_func(c)) for c in all_currents])
        res_freq_lows = np.array([float(low_power_res_fit_func(c)) for c in all_currents])

        self.writer.add_data(
            current=all_currents,
            gen_power=all_gen_powers,

            base_resonant_frequency=res_freq_refs,
            low_power_res_frequency=res_freq_lows,

            driven_resonant_frequency=popts_arr[:, 2] / (2 * np.pi),
            driven_Qint=popts_arr[:, 1],
            driven_Qext=popts_arr[:, 0],

            driven_resonant_frequency_error=np.array([np.sqrt(p[2, 2]) for p in pconvs_arr]) / (2 * np.pi),
            driven_Qint_error=np.array([np.sqrt(p[1, 1]) for p in pconvs_arr]),
            driven_Qext_error=np.array([np.sqrt(p[0, 0]) for p in pconvs_arr]),

            res_shift_ref_undriven=popts_arr[:, 2] / (2 * np.pi) - res_freq_refs,
            res_shift_ref_low=popts_arr[:, 2] / (2 * np.pi) - res_freq_lows
        )
    def close_file(self):
        self.writer.__exit__(None, None, None)
            
    def single_fit(self, vna_freqs, phase_trace, pow_trace, f0Guess, QextGuess = 50, QintGuess = 300, magBackGuess = 0.0001, bounds = None, smooth = False, smooth_win = 11, phaseOffGuess = 0, debug = False, adaptive_window = False, adapt_win_size = 300e6): 
        f0Guess = f0Guess*2*np.pi
        if bounds == None: 
            bounds=self.default_bounds(QextGuess, QintGuess, f0Guess, magBackGuess)
            
        if adaptive_window: 
            filt1 = vna_freqs < f0Guess + adapt_win_size*2*np.pi
            filt2 = vna_freqs > f0Guess - adapt_win_size*2*np.pi
            filt = filt1*filt2
            vna_freqs = np.copy(vna_freqs[filt])
            phase_trace = np.copy(phase_trace[filt])
            pow_trace = np.copy(pow_trace[filt])
            
            
        if debug: 
            plt.figure(1)
            plt.plot(vna_freqs/(2*np.pi), phase_trace)
            plt.title("Debug1: phase")
            plt.figure(2)
            plt.plot(vna_freqs/(2*np.pi), pow_trace)
            plt.title("Debug1: power")
            
        if smooth: 
            phase_trace = savgol_filter(phase_trace, smooth_win, 3)
            pow_trace = savgol_filter(pow_trace, smooth_win, 3)
            
        lin = 10**(pow_trace/20)
        
        imag = lin * np.sin(phase_trace)
        real = lin * np.cos(phase_trace)

        popt, pconv = fit(vna_freqs, real, imag, pow_trace, phase_trace, Qguess = (QextGuess,QintGuess), f0Guess = f0Guess, real_only = 0, bounds = bounds, magBackGuess = magBackGuess, phaseGuess = phaseOffGuess)
        
        print(f'f (Hz): {np.round(popt[2]/2/np.pi, 3)}', )
        fitting_params = list(inspect.signature(reflectionFunc).parameters.keys())[1:]
        for i in range(2):
            print(f'{fitting_params[i]}: {np.round(popt[i], 2)} +- {np.round(np.sqrt(pconv[i, i]), 3)}')
        Qtot = popt[0] * popt[1] / (popt[0] + popt[1])
        print('Q_tot: ', round(Qtot), '\nT1 (s):', round(Qtot/popt[2]), f"Kappa: {round(popt[2]/2/np.pi/Qtot)}", )
        
        self.initial_popt = popt
        self.initial_pconv = pconv
        
        if debug: 
            plotRes(vna_freqs, real, imag, pow_trace, phase_trace, popt)
            
        return(popt, pconv)

    def initial_fit(self,
                    f0Guess,
                    QextGuess=50, QintGuess=300, magBackGuess=0.0001,
                    bounds=None, smooth=False, smooth_win=11,
                    phaseOffGuess=0, debug=False,
                    adaptive_window=False, adapt_win_size=300e6,
                    elec_delay=None, delay_unit='ns',
                    delay_sign=+1, delay_reference='mean'):
        """
        Fit the first (current, gen_power) slice to get initial_popt/pconv.

        elec_delay      : float or None.  Delay value in delay_unit to remove before fitting.
        delay_unit      : 's', 'ns', or 'ps'
        delay_sign      : +1  →  phi += sign*(omega-omega_ref)*tau
        delay_reference : 'zero', 'mean', or a reference frequency in Hz
        """
        if bounds is None:
            bounds = self.default_bounds(QextGuess, QintGuess, f0Guess * 2 * np.pi, magBackGuess)

        filt = ((self.currents == np.unique(self.currents)[0]) &
                (self.gen_powers == np.unique(self.gen_powers)[0]))

        init_vna_freqs = self.vna_freqs[filt]  # rad/s
        init_phase = self.undriven_vna_phase[filt].copy()
        init_pow_trace = self.undriven_vna_power[filt]

        # --- electrical delay correction ---
        if elec_delay is not None:
            init_phase = self._apply_elec_delay(
                init_vna_freqs, init_phase,
                elec_delay, delay_unit, delay_sign, delay_reference
            )
            print(f"[initial_fit] Applied elec_delay={elec_delay} {delay_unit} "
                  f"(sign={delay_sign:+d}, ref={delay_reference})")

        if debug:
            plt.figure(1)
            plt.plot(init_vna_freqs / (2 * np.pi), init_phase)
            plt.title("Debug: phase after delay correction")
            plt.figure(2)
            plt.plot(init_vna_freqs / (2 * np.pi), init_pow_trace)
            plt.title("Debug: power")

        lin = 10 ** (init_pow_trace / 20)
        real = lin * np.cos(init_phase)
        imag = lin * np.sin(init_phase)

        popt, pconv = self.single_fit(
            init_vna_freqs, init_phase, init_pow_trace,
            f0Guess,
            QintGuess=QintGuess, QextGuess=QextGuess,
            magBackGuess=magBackGuess, phaseOffGuess=phaseOffGuess,
            adaptive_window=adaptive_window, adapt_win_size=adapt_win_size,
            debug=debug
        )
        self.initial_popt = popt
        self.initial_pconv = pconv

        plotRes(init_vna_freqs, real, imag, init_pow_trace, init_phase, popt)

    def _safe_fit(self, freq, real, imag_, mag, phase,
                  QextG, QintG, f0G, magBackG, iter_bounds):
        try:
            return fit(freq, real, imag_, mag, phase,
                        Qguess=(QextG, QintG), f0Guess=f0G,
                        real_only=0, bounds=iter_bounds, magBackGuess=magBackG)
        except (RuntimeError, ValueError) as e:
            print(f"  [_safe_fit] FAILED ({e.__class__.__name__}: {e}) — marking NaN.")
            bad_popt = np.full(5, np.nan)
            bad_pconv = np.full((5, 5), np.inf)
            return bad_popt, bad_pconv

    def semiauto_fit(self, bias_currents, vna_freqs, vna_mags, vna_phases, popt,
                     debug=False, smooth=False, smooth_win=11,
                     adaptive_window=False, adapt_win_size=300e6,
                     fourier_filter=False, fourier_cutoff=40,
                     pconv_tol=10, bounds=None, accept_low_conv=False,
                     elec_delay=None, delay_unit='ns',
                     delay_sign=+1, delay_reference='mean',
                     qint_upper_multiplier=5,
                     pconv_explosion_threshold=1e6):
        """
        Semi-automatic sweep fit over all bias currents at a fixed gen_power.

        elec_delay      : float or None.  Applied per-slice before fitting.
        delay_unit      : 's', 'ns', or 'ps'
        delay_sign      : +1  →  phi += sign*(omega-omega_ref)*tau
        delay_reference : 'zero', 'mean', or a reference frequency in Hz
        """
        n_currents = np.size(np.unique(bias_currents))
        res_freqs = np.zeros(n_currents)
        Qints = np.zeros(n_currents)
        Qexts = np.zeros(n_currents)
        magBacks = np.zeros(n_currents)
        popts, pconvs = [], []

        init_f0 = popt[2]
        init_Qint = popt[1]
        init_Qext = popt[0]
        init_magBack = popt[3]
        last_good_popt = popt.copy()
        last_good_popt_valid = False          # ← NEW: tracks whether we ever got a real good fit
        prev_pconv = np.zeros((5, 5))

        MAX_REASONABLE_Q = 1e9               # ← NEW: sanity cap on Q values

        for i, current in enumerate(np.sort(np.unique(bias_currents))):
            print(current)
            mask = bias_currents == current
            slice_omega = vna_freqs[mask] * 2 * np.pi
            slice_phase = vna_phases[mask].copy()
            slice_mag_db = vna_mags[mask]
            slice_mag = 10 ** (slice_mag_db / 20)

            if elec_delay is not None:
                slice_phase = self._apply_elec_delay(
                    slice_omega, slice_phase,
                    elec_delay, delay_unit, delay_sign, delay_reference
                )

            if smooth:
                slice_phase = savgol_filter(slice_phase, smooth_win, 3)
                slice_mag = savgol_filter(slice_mag, smooth_win, 3)

            real = slice_mag * np.cos(slice_phase)
            imag_ = slice_mag * np.sin(slice_phase)

            if fourier_filter:
                real = idct(dct(real)[fourier_cutoff:])
                imag_ = idct(dct(imag_)[fourier_cutoff:])

            # --- guesses ---
            if i >= 2:
                good = res_freqs[:i][np.isfinite(res_freqs[:i])]
                f0Guess = (np.average(good[-1:]) if len(good) else init_f0 / (2 * np.pi)) * 2 * np.pi

                # ── detect exploded Qint from previous slice and reset ──────────
                qint_var = pconv[1, 1] if np.isfinite(pconv[1, 1]) else np.inf
                if qint_var > pconv_explosion_threshold:
                    print(f"  [semiauto_fit] Qint covariance exploded "
                          f"(pconv[1,1]={qint_var:.2e}) at slice {i - 1} — "
                          f"resetting Q guesses to last known good.")

                    # ── NEW: validate last_good_popt before trusting it ─────────
                    lgp = last_good_popt
                    lgp_ok = (last_good_popt_valid
                              and np.isfinite(lgp).all()
                              and lgp[0] < MAX_REASONABLE_Q
                              and lgp[1] < MAX_REASONABLE_Q)

                    if lgp_ok:
                        QextGuess   = lgp[0]
                        QintGuess   = lgp[1]
                        magBackGuess = lgp[3]
                        print(f"  [semiauto_fit] last_good_popt: "
                              f"Qext={lgp[0]:.2f}, Qint={lgp[1]:.2f}, "
                              f"f0={lgp[2]/2/np.pi:.6e} Hz, magBack={lgp[3]:.4f}")
                    else:
                        print(f"  [semiauto_fit] last_good_popt unusable "
                              f"(valid={last_good_popt_valid}, values={lgp}) — "
                              f"falling back to init guesses.")
                        QextGuess    = init_Qext
                        QintGuess    = init_Qint
                        magBackGuess = init_magBack

                    print(f"  [semiauto_fit] bounds will be: "
                          f"Qext=[{QextGuess / 1.5:.2f}, {QextGuess * 1.5:.2f}], "
                          f"Qint=[{10000}, {QintGuess * qint_upper_multiplier:.2f}], "
                          f"f0=[{(f0_anchor - 1000e6 * 2 * np.pi) / 2 / np.pi:.6e}, {(f0_anchor + 1000e6 * 2 * np.pi) / 2 / np.pi:.6e}], "
                          f"magBack=[{magBackGuess / 2:.4f}, {magBackGuess * 2:.4f}], "
                          f"phase=[-2pi, +2pi]")

                else:
                    magBackGuess = np.average(magBacks[i - 1:i])
                    QextGuess    = np.average(Qexts[i - 1:i])
                    QintGuess    = np.average(Qints[i - 1:i])

                    if not np.isfinite(QextGuess):    QextGuess    = last_good_popt[0] if last_good_popt_valid else init_Qext
                    if not np.isfinite(QintGuess):    QintGuess    = last_good_popt[1] if last_good_popt_valid else init_Qint
                    if not np.isfinite(magBackGuess): magBackGuess = last_good_popt[3] if last_good_popt_valid else init_magBack

                if debug:
                    print('f0Guess = ', f0Guess)
                    print('magBackGuess = ', magBackGuess)
                    print('QextGuess = ', QextGuess)
                    print('QintGuess = ', QintGuess)

                try:
                    f0_anchor = self.res_func(current) * 2 * np.pi
                except Exception:
                    f0_anchor = f0Guess

                if adaptive_window:
                    hw = adapt_win_size * 2 * np.pi
                    filt = (slice_omega < f0Guess + hw) & (slice_omega > f0Guess - hw)
                    if np.sum(filt) == 0:
                        print(f"  [semiauto_fit] WARNING: adaptive window empty at "
                              f"current={current:.6f} A — falling back to full range.")
                        filt = np.ones(slice_omega.size, dtype=bool)
                else:
                    filt = np.ones(slice_omega.size, dtype=bool)
            else:
                f0Guess      = init_f0
                magBackGuess = init_magBack
                QextGuess    = init_Qext
                QintGuess    = init_Qint
                f0_anchor    = f0Guess
                filt         = np.ones(slice_omega.size, dtype=bool)

            iter_bounds = (
                [QextGuess / 1.5, 10000,
                 f0_anchor - 1000e6 * 2 * np.pi, magBackGuess / 2, -2 * np.pi],
                [QextGuess * 1.5, QintGuess * qint_upper_multiplier,
                 f0_anchor + 1000e6 * 2 * np.pi, magBackGuess * 2, +2 * np.pi]
            ) if bounds is None else bounds

            if i > 2:
                prev_pconv = pconv

            popt_cur, pconv = self._safe_fit(
                slice_omega[filt], real[filt], imag_[filt],
                slice_mag, slice_phase,
                QextGuess, QintGuess, f0Guess, magBackGuess, iter_bounds
            )
            fit_failed = (not np.isfinite(popt_cur[2]))

            if not fit_failed and np.isfinite(pconv[1, 1]) and pconv[1, 1] < pconv_explosion_threshold:
                # ── NEW: also check Q values are physically reasonable ─────────
                if popt_cur[0] < MAX_REASONABLE_Q and popt_cur[1] < MAX_REASONABLE_Q:
                    last_good_popt = popt_cur.copy()
                    last_good_popt_valid = True

            # --- pconv stability check ---
            if i > 2 and not fit_failed and not np.any(np.isinf(prev_pconv)):
                pconv_diff_ratio = (
                    np.array([pconv[0, 0], pconv[1, 1]]) -
                    np.array([prev_pconv[0, 0], prev_pconv[1, 1]])
                ) / np.array([prev_pconv[0, 0], prev_pconv[1, 1]])

                if debug:
                    print(f"  pconv_diff_ratio: {pconv_diff_ratio}")

                alt_offsets = np.array([
                    1, -1, 5, -5, 10, -10, 15, -15, 20, -20,
                    30, -30, 50, -50, 100, -100, 300, -300, 500, -500, 750, -750, 1000, -1000, 1500, -1500, 2000, -2000
                ]) * 1e6 * 2 * np.pi

                j = 0
                while np.any(np.abs(pconv_diff_ratio) > pconv_tol):
                    if j > len(alt_offsets) - 1:
                        if accept_low_conv:
                            print(f"  [semiauto_fit] Exhausted retries at (bias={current}, "
                                  f"power={self.latest_power}) — accepting low convergence.")
                        else:
                            print(f"  [semiauto_fit] Exhausted retries at (bias={current}, "
                                  f"power={self.latest_power}) — saving NaN.")
                            popt_cur = np.full(5, np.nan)
                            pconv = np.full((5, 5), np.inf)
                            fit_failed = True
                        break

                    print(f"  Sudden Q change (ratio={pconv_diff_ratio}), "
                          f"retrying f0 + {alt_offsets[j] / (2 * np.pi) / 1e6:.0f} MHz")
                    retry_bounds = self.default_bounds(
                        QextGuess, QintGuess, f0Guess + alt_offsets[j], magBackGuess)
                    popt_cur, pconv = self._safe_fit(
                        slice_omega[filt], real[filt], imag_[filt],
                        slice_mag, slice_phase,
                        QextGuess, QintGuess, f0Guess + alt_offsets[j],
                        magBackGuess, retry_bounds
                    )

                    if not np.isfinite(popt_cur[2]):
                        fit_failed = True
                        break

                    pconv_diff_ratio = (
                        np.array([pconv[0, 0], pconv[1, 1]]) -
                        np.array([prev_pconv[0, 0], prev_pconv[1, 1]])
                    ) / np.array([prev_pconv[0, 0], prev_pconv[1, 1]])
                    j += 1

            if debug and not fit_failed:
                import time
                plotRes(slice_omega[filt], real[filt], imag_[filt],
                        slice_mag[filt], slice_phase[filt], popt_cur)
                time.sleep(1)

            if fit_failed:
                print(f"  [semiauto_fit] Skipping debug plot for failed slice "
                      f"(current={current:.6f} A).")

            res_freqs[i] = popt_cur[2] / (2 * np.pi)
            Qints[i]     = popt_cur[1]
            Qexts[i]     = popt_cur[0]
            magBacks[i]  = popt_cur[3]
            popts.append(popt_cur)
            pconvs.append(pconv)

            if not fit_failed:
                fitting_params = list(inspect.signature(reflectionFunc).parameters.keys())[1:]
                Qtot = popt_cur[0] * popt_cur[1] / (popt_cur[0] + popt_cur[1])
                print(f'  f (Hz): {np.round(popt_cur[2] / 2 / np.pi, 3)}')
                for k in range(2):
                    print(f'  {fitting_params[k]}: {np.round(popt_cur[k], 2)} '
                          f'+- {np.round(np.sqrt(pconv[k, k]), 3)}')
                print(f'  Q_tot: {round(Qtot)}  T1: {round(Qtot / popt_cur[2])} s  '
                      f'Kappa: {round(popt_cur[2] / 2 / np.pi / Qtot)} Hz')

        return np.unique(bias_currents), res_freqs, Qints, Qexts, magBacks, popts, pconvs
        
    def default_bounds(self, QextGuess, QintGuess, f0Guess, magBackGuess):
        return ([QextGuess / 1.5, QintGuess / 1.5, f0Guess - 500e6 * 2 * np.pi, magBackGuess / 2, -2*np.pi],
                [QextGuess * 1.5, QintGuess *3, f0Guess + 500e6 * 2 * np.pi, magBackGuess * 2, 2*np.pi])

    def fit(self,
            debug=False,
            save_data=False,
            max_gen_power=None,
            smooth=False,
            smooth_win=11,
            adaptive_window=False,
            adapt_win_size=300e6,
            bounds=None,
            fourier_filter=False,
            fourier_cutoff=40,
            pconv_tol=10,
            accept_low_conv=False,
            elec_delay=None,
            delay_unit='ns',
            delay_sign=+1,
            delay_reference='mean'):

        all_currents = []
        all_gen_powers = []
        all_popts = []
        all_pconvs = []

        if max_gen_power is not None:
            fitted_gen_powers = np.unique(self.gen_powers) <= max_gen_power
        else:
            fitted_gen_powers = np.unique(self.gen_powers) <= np.max(np.unique(self.gen_powers))

        try:
            for i, gen_power in enumerate(np.unique(self.gen_powers)[fitted_gen_powers]):
                print(gen_power)
                self.latest_power = gen_power
                pow_condn = self.gen_powers == gen_power

                bias_currents = self.currents[pow_condn]
                vna_freqs = self.vna_freqs[pow_condn]
                vna_phases = self.driven_vna_phase[pow_condn]
                vna_mags = self.driven_vna_power[pow_condn]
                print(f"Generator Power: {gen_power} dBm")

                fit_currents, fit_freqs, fit_Qints, fit_Qexts, fit_magBacks, popts, pconvs = self.semiauto_fit(
                    bias_currents,
                    vna_freqs / (2 * np.pi),  # semiauto_fit expects Hz, multiplies by 2pi internally
                    vna_mags,
                    vna_phases,
                    self.initial_popt,
                    debug=debug,
                    smooth=smooth,
                    smooth_win=smooth_win,
                    adaptive_window=adaptive_window,
                    adapt_win_size=adapt_win_size,
                    fourier_filter=fourier_filter,
                    fourier_cutoff=fourier_cutoff,
                    pconv_tol=pconv_tol,
                    bounds=bounds,
                    accept_low_conv=accept_low_conv,
                    elec_delay=elec_delay,
                    delay_unit=delay_unit,
                    delay_sign=delay_sign,
                    delay_reference=delay_reference,
                )

                if i == 0:
                    self.low_power_res_fit_func = interp1d(fit_currents, fit_freqs, 'linear')

                if save_data:
                    all_currents.append(fit_currents)
                    all_gen_powers.append(np.full_like(fit_currents, gen_power))
                    all_popts.extend(popts)
                    all_pconvs.extend(pconvs)

            if save_data:
                self.save_fit_batched(
                    np.concatenate(all_currents),
                    np.concatenate(all_gen_powers),
                    all_popts,
                    all_pconvs,
                    self.low_power_res_fit_func,
                )
        finally:
            if save_data:
                self.close_file()

    def _apply_elec_delay(self, omega, phase, elec_delay, delay_unit='ns', delay_sign=+1, delay_reference='mean'):
        """
        Internal helper: apply electrical delay correction to a phase array.
        omega  : rad/s array (same length as phase)
        phase  : phase array to correct (not modified in-place; returns corrected copy)
        Returns corrected phase array.
        """
        unit_map = {'s': 1.0, 'ns': 1e-9, 'ps': 1e-12}
        if delay_unit not in unit_map:
            raise ValueError("delay_unit must be 's', 'ns', or 'ps'")
        tau = float(elec_delay) * unit_map[delay_unit]

        if delay_reference == 'zero':
            omega_ref = 0.0
        elif delay_reference == 'mean':
            omega_ref = float(np.mean(omega))
        else:
            omega_ref = float(delay_reference) * 2 * np.pi

        return phase + delay_sign * (omega - omega_ref) * tau

    def plot_phase_heatmap_interactive(self, data='driven', figsize=(9, 6), cmap='RdYlBu_r',
                                       freq_unit='GHz', vmin=None, vmax=None,
                                       elec_delay=None, delay_unit='ns',
                                       delay_sign=+1, delay_reference='mean',
                                       save_gif=False, gif_path='phase_vs_current.gif', gif_fps=5,
                                       slider_axis='current'):  # ← 'current' or 'power'

        import numpy as np
        import matplotlib.pyplot as plt
        import traceback
        from matplotlib.widgets import Slider
        from matplotlib.animation import PillowWriter

        if slider_axis not in ('current', 'power'):
            raise ValueError("slider_axis must be 'current' or 'power'")

        freq_scales = {'GHz': 1e9, 'MHz': 1e6, 'Hz': 1.0}
        if freq_unit not in freq_scales:
            raise ValueError("freq_unit must be 'GHz', 'MHz', or 'Hz'")
        fscale = freq_scales[freq_unit]

        if data == 'driven':
            phase_all = self.driven_vna_phase
            data_label = 'Driven'
        elif data == 'undriven':
            phase_all = self.undriven_vna_phase
            data_label = 'Undriven'
        else:
            raise ValueError("data must be 'driven' or 'undriven'")

        unique_currents, inv_c = np.unique(self.currents, return_inverse=True)
        unique_powers, inv_p = np.unique(self.gen_powers, return_inverse=True)

        if len(unique_currents) == 0:
            raise ValueError("No current data found.")

        vmin_plot = vmin if vmin is not None else -np.pi
        vmax_plot = vmax if vmax is not None else +np.pi

        def centers_to_edges(x):
            x = np.asarray(x, dtype=float)
            if len(x) == 1:
                return np.array([x[0] - 0.5, x[0] + 0.5], dtype=float)
            dx = np.diff(x)
            edges = np.empty(len(x) + 1, dtype=float)
            edges[1:-1] = x[:-1] + dx / 2
            edges[0] = x[0] - dx[0] / 2
            edges[-1] = x[-1] + dx[-1] / 2
            return edges

        # ── slider over current, y-axis = power ──────────────────────────────────
        def get_grid_by_current(i_current):
            mask = (inv_c == i_current)
            freqs = self.vna_freqs[mask] / (2 * np.pi)
            omega = self.vna_freqs[mask]
            powers = self.gen_powers[mask]
            phase = phase_all[mask].copy()

            if elec_delay is not None:
                phase = self._apply_elec_delay(
                    omega, phase, elec_delay, delay_unit, delay_sign, delay_reference)

            uf = np.sort(np.unique(freqs))
            up = np.sort(np.unique(powers))
            fi = np.searchsorted(uf, freqs)
            pi = np.searchsorted(up, powers)

            grid = np.full((len(up), len(uf)), np.nan)
            grid[pi, fi] = phase

            X, Y = np.meshgrid(centers_to_edges(uf / fscale), centers_to_edges(up))
            return X, Y, grid

        # ── slider over power, y-axis = current ──────────────────────────────────
        def get_grid_by_power(i_power):
            mask = (inv_p == i_power)
            freqs = self.vna_freqs[mask] / (2 * np.pi)
            omega = self.vna_freqs[mask]
            currents = self.currents[mask]
            phase = phase_all[mask].copy()

            if elec_delay is not None:
                phase = self._apply_elec_delay(
                    omega, phase, elec_delay, delay_unit, delay_sign, delay_reference)

            uf = np.sort(np.unique(freqs))
            uc = np.sort(np.unique(currents))
            fi = np.searchsorted(uf, freqs)
            ci = np.searchsorted(uc, currents)

            grid = np.full((len(uc), len(uf)), np.nan)
            grid[ci, fi] = phase

            X, Y = np.meshgrid(centers_to_edges(uf / fscale), centers_to_edges(uc))
            return X, Y, grid

        # ── pick mode ─────────────────────────────────────────────────────────────
        if slider_axis == 'current':
            n_slider = len(unique_currents)
            get_grid = get_grid_by_current
            slider_lbl = 'Current idx'
            y_label = 'Generator power (dBm)'

            def slider_title(idx):
                return (f'{data_label} phase heatmap @ '
                        f'current = {unique_currents[idx]:.7f} A')
        else:
            n_slider = len(unique_powers)
            get_grid = get_grid_by_power
            slider_lbl = 'Power idx'
            y_label = 'Current (A)'

            def slider_title(idx):
                return (f'{data_label} phase heatmap @ '
                        f'power = {unique_powers[idx]:.1f} dBm')

        # ── precompute cache ──────────────────────────────────────────────────────
        print("Precomputing cache …")
        cache = {}
        for i in range(n_slider):
            cache[i] = get_grid(i)
        print("Done.")

        # ── initial plot ──────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=figsize)
        plt.subplots_adjust(bottom=0.18)

        X, Y, grid = cache[0]
        mesh = ax.pcolormesh(X, Y, grid, cmap=cmap,
                             vmin=vmin_plot, vmax=vmax_plot, shading='flat')

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label('Phase (rad)')
        ax.set_xlabel(f'VNA frequency ({freq_unit})')
        ax.set_ylabel(y_label)
        title = ax.set_title(slider_title(0))

        # ── slider ────────────────────────────────────────────────────────────────
        ax_slider = plt.axes([0.15, 0.05, 0.70, 0.03])
        slider = Slider(ax=ax_slider, label=slider_lbl,
                        valmin=0, valmax=n_slider - 1,
                        valinit=0, valstep=1)

        state = {'mesh': mesh}

        def update_idx(idx):
            idx = int(idx)
            X, Y, grid = cache[idx]
            state['mesh'].remove()
            state['mesh'] = ax.pcolormesh(X, Y, grid, cmap=cmap,
                                          vmin=vmin_plot, vmax=vmax_plot,
                                          shading='flat')
            cbar.update_normal(state['mesh'])
            title.set_text(slider_title(idx))
            ax.set_xlim(X[0, 0], X[0, -1])
            ax.set_ylim(Y[0, 0], Y[-1, 0])
            fig.canvas.draw_idle()

        def slider_update(val):
            try:
                update_idx(val)
            except Exception:
                traceback.print_exc()

        slider.on_changed(slider_update)

        # keep references alive
        self._phase_fig = fig
        self._phase_ax = ax
        self._phase_slider = slider
        self._phase_slider_update = slider_update

        # ── GIF export ────────────────────────────────────────────────────────────
        if save_gif:
            writer = PillowWriter(fps=gif_fps)
            with writer.saving(fig, gif_path, dpi=120):
                for i in range(n_slider):
                    update_idx(i)
                    writer.grab_frame()
            print(f"GIF saved to: {gif_path}")

        plt.show()

    def plot_single_slice(self, target_current, target_power,
                          freq_unit='GHz', elec_delay=None, delay_unit='ns',
                          delay_sign=+1, delay_reference='mean',
                          smooth=False, smooth_win=11,
                          figsize=(12, 4)):
        """
        Plot the raw VNA trace (phase, magnitude, IQ) for the single
        (current, gen_power) point closest to (target_current, target_power).

        Parameters
        ----------
        target_current : float   Target bias current in Amps
        target_power   : float   Target generator power in dBm
        freq_unit      : str     'GHz', 'MHz', or 'Hz'
        elec_delay     : float or None   Delay to remove before plotting
        delay_unit     : str     'ns', 'ps', or 's'
        delay_sign     : int     +1 or -1
        delay_reference: str/float  'zero', 'mean', or reference freq in Hz
        smooth         : bool    Apply Savitzky-Golay smoothing
        smooth_win     : int     Smoothing window (must be odd)
        figsize        : tuple
        """
        freq_scales = {'GHz': 1e9, 'MHz': 1e6, 'Hz': 1.0}
        if freq_unit not in freq_scales:
            raise ValueError("freq_unit must be 'GHz', 'MHz', or 'Hz'")
        fscale = freq_scales[freq_unit]

        # ── find closest (current, power) point ─────────────────────────────────
        unique_currents = np.sort(np.unique(self.currents))
        unique_powers = np.sort(np.unique(self.gen_powers))

        closest_current = unique_currents[np.argmin(np.abs(unique_currents - target_current))]
        closest_power = unique_powers[np.argmin(np.abs(unique_powers - target_power))]

        mask = (self.currents == closest_current) & (self.gen_powers == closest_power)

        if not np.any(mask):
            raise ValueError(f"No data found for current={closest_current}, power={closest_power}")

        omega = self.vna_freqs[mask]  # rad/s
        freq = omega / (2 * np.pi) / fscale  # display units
        phase = self.driven_vna_phase[mask].copy()
        mag_db = self.driven_vna_power[mask]

        print(f"Closest point found:")
        print(f"  current = {closest_current:.7f} A  (requested {target_current:.7f} A)")
        print(f"  power   = {closest_power} dBm       (requested {target_power} dBm)")
        print(f"  points  = {np.sum(mask)}")

        # ── optional delay correction ────────────────────────────────────────────
        if elec_delay is not None:
            phase = self._apply_elec_delay(
                omega, phase, elec_delay, delay_unit, delay_sign, delay_reference
            )
            delay_str = f', delay={elec_delay} {delay_unit}'
        else:
            delay_str = ''

        # ── optional smoothing ───────────────────────────────────────────────────
        if smooth:
            from scipy.signal import savgol_filter
            phase = savgol_filter(phase, smooth_win, 3)
            mag_db = savgol_filter(mag_db, smooth_win, 3)

        lin = 10 ** (mag_db / 20)
        real = lin * np.cos(phase)
        imag = lin * np.sin(phase)

        # ── plot ─────────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(
            f'Slice: current = {closest_current:.7f} A,  power = {closest_power} dBm{delay_str}',
            fontsize=11
        )

        # Phase
        axes[0].plot(freq, np.unwrap(phase), color='steelblue', lw=1.4)
        axes[0].set_xlabel(f'Frequency ({freq_unit})')
        axes[0].set_ylabel('Phase (rad)')
        axes[0].set_title('Unwrapped Phase')
        axes[0].grid(True, alpha=0.3)

        # Magnitude
        axes[1].plot(freq, mag_db, color='darkorange', lw=1.4)
        axes[1].set_xlabel(f'Frequency ({freq_unit})')
        axes[1].set_ylabel('Magnitude (dBm)')
        axes[1].set_title('Magnitude')
        axes[1].grid(True, alpha=0.3)

        # IQ loop
        axes[2].plot(real, imag, color='forestgreen', lw=1.4)
        axes[2].plot(real[0], imag[0], 'o', color='blue', ms=6, label='start')
        axes[2].plot(real[-1], imag[-1], 's', color='red', ms=6, label='end')
        axes[2].set_xlabel('Re')
        axes[2].set_ylabel('Im')
        axes[2].set_title('IQ Loop')
        axes[2].set_aspect('equal', 'datalim')
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        return closest_current, closest_power

    def test_semiauto_fit(self, target_power,
                          debug=False,
                          smooth=False, smooth_win=11,
                          adaptive_window=False, adapt_win_size=300e6,
                          fourier_filter=False, fourier_cutoff=40,
                          pconv_tol=10, bounds=None, accept_low_conv=False,
                          elec_delay=None, delay_unit='ns',
                          delay_sign=+1, delay_reference='mean',
                          freq_unit='GHz',
                          figsize=(14, 5),
                          pconv_explosion_threshold=1e6):

        self.latest_power = target_power
        freq_scales = {'GHz': 1e9, 'MHz': 1e6, 'Hz': 1.0}
        if freq_unit not in freq_scales:
            raise ValueError("freq_unit must be 'GHz', 'MHz', or 'Hz'")
        fscale = freq_scales[freq_unit]

        # ── find closest power ────────────────────────────────────────────────
        unique_powers = np.sort(np.unique(self.gen_powers))
        closest_power = unique_powers[np.argmin(np.abs(unique_powers - target_power))]
        print(f"Target power: {target_power} dBm  →  using: {closest_power} dBm")

        # ── slice data for this power ─────────────────────────────────────────
        mask = self.gen_powers == closest_power
        bias_curr = self.currents[mask]
        vna_freqs = self.vna_freqs[mask]
        vna_phases = self.driven_vna_phase[mask]
        vna_mags = self.driven_vna_power[mask]

        print(f"Number of current points: {len(np.unique(bias_curr))}")
        print(f"Using initial_popt: {self.initial_popt}")

        # ── run semiauto_fit ──────────────────────────────────────────────────
        fit_currents, fit_freqs, fit_Qints, fit_Qexts, fit_magBacks, popts, pconvs = \
            self.semiauto_fit(
                bias_currents=bias_curr,
                vna_freqs=vna_freqs / (2 * np.pi),
                vna_mags=vna_mags,
                vna_phases=vna_phases,
                popt=self.initial_popt,
                debug=debug,
                smooth=smooth,
                smooth_win=smooth_win,
                adaptive_window=adaptive_window,
                adapt_win_size=adapt_win_size,
                fourier_filter=fourier_filter,
                fourier_cutoff=fourier_cutoff,
                pconv_tol=pconv_tol,
                bounds=bounds,
                accept_low_conv=accept_low_conv,
                elec_delay=elec_delay,
                delay_unit=delay_unit,
                delay_sign=delay_sign,
                delay_reference=delay_reference,
                pconv_explosion_threshold=pconv_explosion_threshold,

            )

        # ── extract results ───────────────────────────────────────────────────
        popts_arr = np.array(popts)
        pconvs_arr = np.array(pconvs)

        f0_hz = popts_arr[:, 2] / (2 * np.pi)
        Qext_arr = popts_arr[:, 0]
        Qint_arr = popts_arr[:, 1]

        f0_err = np.array([np.sqrt(p[2, 2]) / (2 * np.pi)
                           if np.isfinite(p[2, 2]) else np.nan
                           for p in pconvs_arr])
        Qext_err = np.array([np.sqrt(p[0, 0])
                             if np.isfinite(p[0, 0]) else np.nan
                             for p in pconvs_arr])
        Qint_err = np.array([np.sqrt(p[1, 1])
                             if np.isfinite(p[1, 1]) else np.nan
                             for p in pconvs_arr])

        # flux sweep reference
        try:
            f0_ref = np.array([self.res_func(c) for c in fit_currents])
            has_ref = True
        except Exception:
            has_ref = False
            print("  (res_func not available — skipping flux sweep overlay)")

        # ── flag bad fits ─────────────────────────────────────────────────────
        bad_mask = ~np.isfinite(f0_hz)
        n_bad = np.sum(bad_mask)
        n_total = len(fit_currents)
        print(f"\nFit results: {n_total - n_bad}/{n_total} converged, "
              f"{n_bad} failed (NaN)")

        if has_ref:
            deviation_hz = np.abs(f0_hz - f0_ref)
            large_dev = np.isfinite(deviation_hz) & (deviation_hz > 300e6)
            print(f"Fits deviating >300 MHz from flux sweep: {np.sum(large_dev)}")

        unique_currents = np.sort(np.unique(bias_curr))
        n_currents = len(unique_currents)

        # ══════════════════════════════════════════════════════════════════════
        # FIGURE 1 — phase heatmap + fitted f0, shared current y-axis
        # ══════════════════════════════════════════════════════════════════════

        # build phase grid (current × freq)
        unique_freqs_all = np.sort(np.unique(vna_freqs / (2 * np.pi)))

        fi = np.searchsorted(unique_freqs_all, vna_freqs / (2 * np.pi))
        ci = np.searchsorted(unique_currents, bias_curr)

        phase_corrected = vna_phases.copy()
        if elec_delay is not None:
            phase_corrected = self._apply_elec_delay(
                vna_freqs, phase_corrected,
                elec_delay, delay_unit, delay_sign, delay_reference
            )

        phase_grid = np.full((len(unique_currents), len(unique_freqs_all)), np.nan)
        phase_grid[ci, fi] = phase_corrected

        def _edges(x):
            x = np.asarray(x, dtype=float)
            if len(x) == 1:
                return np.array([x[0] - 0.5, x[0] + 0.5])
            dx = np.diff(x)
            e = np.empty(len(x) + 1)
            e[1:-1] = x[:-1] + dx / 2
            e[0] = x[0] - dx[0] / 2
            e[-1] = x[-1] + dx[-1] / 2
            return e

        freq_edges = _edges(unique_freqs_all / fscale)
        current_edges = _edges(unique_currents)
        F, C = np.meshgrid(freq_edges, current_edges)

        fig1, (ax_f0, ax_map) = plt.subplots(
            1, 2, figsize=(14, 6),
            sharey=True,
            gridspec_kw={'width_ratios': [1, 2]}
        )
        fig1.suptitle(
            f'Phase heatmap + fitted f0  |  power = {closest_power} dBm',
            fontsize=12, fontweight='bold'
        )

        # ── right: phase heatmap ──────────────────────────────────────────────
        pcm = ax_map.pcolormesh(F, C, phase_grid,
                                cmap='RdYlBu_r', vmin=-np.pi, vmax=np.pi,
                                shading='flat')
        cbar = fig1.colorbar(pcm, ax=ax_map)
        cbar.set_label('Phase (rad)')
        ax_map.set_xlabel(f'VNA frequency ({freq_unit})')
        ax_map.set_title('Phase heatmap')

        good = np.isfinite(f0_hz)
        ax_map.scatter(f0_hz[good] / fscale, fit_currents[good],
                       color='lime', s=8, zorder=5, label='fitted f0')
        if has_ref:
            ax_map.plot(f0_ref / fscale, fit_currents,
                        color='orange', lw=1.2, ls='--', label='flux sweep ref')
        if np.any(bad_mask):
            first_bad_current = fit_currents[np.where(bad_mask)[0][0]]
            ax_map.axhline(first_bad_current, color='red', lw=1,
                           ls='--', label='first NaN')
        ax_map.legend(fontsize=8)

        # ── left: f0 vs current ───────────────────────────────────────────────
        ax_f0.errorbar(f0_hz[good] / fscale, fit_currents[good],
                       xerr=f0_err[good] / fscale,
                       fmt='o', ms=3, lw=1, color='steelblue', label='fitted f0')
        if has_ref:
            ax_f0.plot(f0_ref / fscale, fit_currents,
                       color='darkorange', lw=1.2, ls='--', label='flux sweep ref')
        if np.any(bad_mask):
            ax_f0.axhline(first_bad_current, color='red', lw=1,
                          ls='--', label='first NaN')
        ax_f0.set_xlabel(f'f0 ({freq_unit})')
        ax_f0.set_ylabel('Current (A)')
        ax_f0.set_title('Fitted f0')
        ax_f0.legend(fontsize=8)
        ax_f0.grid(True, alpha=0.3)

        fig1.tight_layout()
        plt.show()

        # # ══════════════════════════════════════════════════════════════════════
        # # FIGURE 2 — Real and Imag vs frequency per slice
        # # ══════════════════════════════════════════════════════════════════════
        # n_cols = min(5, n_currents)
        # n_rows = int(np.ceil(n_currents / n_cols))
        # 
        # fig2, axes2 = plt.subplots(n_rows * 2, n_cols,
        #                            figsize=(4 * n_cols, 3 * n_rows * 2),
        #                            squeeze=False)
        # fig2.suptitle(
        #     f'Real & Imag vs Frequency  |  power = {closest_power} dBm',
        #     fontsize=12, fontweight='bold'
        # )
        # 
        # for idx, current in enumerate(unique_currents):
        #     row, col = divmod(idx, n_cols)
        #     ax_re = axes2[row * 2][col]  # real on even rows
        #     ax_im = axes2[row * 2 + 1][col]  # imag on odd rows
        # 
        #     sl = (bias_curr == current)
        #     omega_sl = vna_freqs[sl]
        #     phase_sl = vna_phases[sl].copy()
        #     mag_sl = 10 ** (vna_mags[sl] / 20)
        # 
        #     if elec_delay is not None:
        #         phase_sl = self._apply_elec_delay(
        #             omega_sl, phase_sl,
        #             elec_delay, delay_unit, delay_sign, delay_reference
        #         )
        # 
        #     sort_idx = np.argsort(omega_sl)
        #     omega_sl = omega_sl[sort_idx]
        #     freq_plot = omega_sl / (2 * np.pi) / fscale  # display units
        #     real_data = (mag_sl * np.cos(phase_sl))[sort_idx]
        #     imag_data = (mag_sl * np.sin(phase_sl))[sort_idx]
        # 
        #     # ── data scatter ──────────────────────────────────────────────────
        #     ax_re.scatter(freq_plot, real_data, s=4, alpha=0.5,
        #                   color='steelblue', zorder=2)
        #     ax_im.scatter(freq_plot, imag_data, s=4, alpha=0.5,
        #                   color='darkorange', zorder=2)
        # 
        #     # ── fit overlay ───────────────────────────────────────────────────
        #     popt_sl = popts[idx]
        #     if np.isfinite(popt_sl[2]):
        #         omega_fine = np.linspace(omega_sl.min(), omega_sl.max(), 500)
        #         freq_fine = omega_fine / (2 * np.pi) / fscale
        # 
        #         # compute fit curve the same way as the data
        #         fit_result = reflectionFunc(omega_fine, *popt_sl)
        # 
        #         # handle both complex output and concatenated [real, imag] output
        #         if np.iscomplexobj(fit_result):
        #             fit_real = fit_result.real
        #             fit_imag = fit_result.imag
        #         else:
        #             # reflectionFunc returns [real_part, imag_part] concatenated
        #             half = len(fit_result) // 2
        #             fit_real = fit_result[:half]
        #             fit_imag = fit_result[half:]
        # 
        #         ax_re.plot(freq_fine, fit_real, color='red', lw=1.5, zorder=3)
        #         ax_im.plot(freq_fine, fit_imag, color='red', lw=1.5, zorder=3)
        # 
        # # hide unused panels (both real and imag rows)
        # for idx in range(n_currents, n_rows * n_cols):
        #     row, col = divmod(idx, n_cols)
        #     axes2[row * 2][col].set_visible(False)
        #     axes2[row * 2 + 1][col].set_visible(False)
        # 
        # fig2.tight_layout()
        # plt.show()
        # 
        # ── print first failure details ───────────────────────────────────────
        if n_bad > 0:
            first_bad_idx = np.where(bad_mask)[0][0]
            print(f"\nFirst failure at index {first_bad_idx}, "
                  f"current = {fit_currents[first_bad_idx] * 1e3:.4f} mA")
            if first_bad_idx > 0:
                print(f"Last good popt: {popts[first_bad_idx - 1]}")
                print(f"Last good f0:   {f0_hz[first_bad_idx - 1] / 1e9:.6f} GHz")
                if has_ref:
                    print(f"Flux sweep f0 there: "
                          f"{f0_ref[first_bad_idx - 1] / 1e9:.6f} GHz")

        return fit_currents, f0_hz, Qext_arr, Qint_arr, popts, pconvs







#%%
    

# #Duffing Autoplot
# #main(Duff_filepath, 'data')


# FS_filepath = r'Z:/Data/SA_2X_B1/fluxsweep/fits/2021-07-22/2021-07-22_0024_SA_2X_B1/2021-07-22_0024_SA_2X_B1.ddh5'
# Duff_filepath = r'Z:/Data/SA_2X_B1/duffing/2021-07-23/2021-07-23_0010_SA_2X_B1_duffing_fine/2021-07-23_0010_SA_2X_B1_duffing_fine.ddh5'
# save_filepath = r'Z:\Data\SA_2X_B1\duffing\fits'

# DFit = fit_Duff_Measurement(Duff_filepath, FS_filepath, save_filepath, 'SA_B1_Duff_fine')
# #%%
# DFit.initial_fit(8.0e9, 
#                   QextGuess = 50, 
#                   QintGuess = 1000, 
#                   magBackGuess = 0.01, 
#                   bounds = None, 
#                   smooth = False, 
#                   smooth_win = 11,
#                   phaseOffGuess = 0, 
#                   debug = False, 
#                   adaptive_window = False, 
#                   adapt_win_size = 300e6
#                 )
# #%%
# print(np.min(DFit.gen_powers))
# #%%
# DFit.fit(
#         debug = False, 
#         save_data = True, 
#         max_gen_power = -20, 
#         savedata = True, 
#         smooth = False, 
#         smooth_win = 11, 
#         adaptive_window = True,  
#         adapt_win_size = 400e6,  
#         fourier_filter = False, 
#         fourier_cutoff = 40, 
#         pconv_tol = 10)
# #%%

    
    
    
    
    
    
    
    
    
    
    