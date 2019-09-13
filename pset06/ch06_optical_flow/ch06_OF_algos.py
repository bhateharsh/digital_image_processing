import numpy as np
import dippykit as dip


def horn_schuck(u, v, dIx, dIy, dIt, param):
    # Set constants
    avg_kernel = np.array([[1/12,  1/6, 1/12],
                           [ 1/6,    0,  1/6],
                           [1/12,  1/6, 1/12]])
    if param is None:
        num_iterations = 100
    else:
        num_iterations = param
    alpha = 1

    for i in range(num_iterations):
        # Iteratively filter the u and v arrays with averaging filters
        u_avg = dip.convolve2d(u, avg_kernel, mode='same')
        v_avg = dip.convolve2d(v, avg_kernel, mode='same')

        # Compute flow vectors constrained by local averages and the optical
        #  flow constaints
        # ============================ EDIT THIS PART =========================
        denominator = np.power(dIx,2) + np.power(dIy,2) + np.power(alpha,2)
        numerator = dIx*u_avg + dIy*v_avg + dIt
        u = u_avg - (dIx*numerator)/denominator
        v = v_avg - (dIy*numerator)/denominator

    # Compute and display the nan and inf ratios
    u_nan_ratio = float(np.sum(np.isnan(u)) / u.size)
    v_nan_ratio = float(np.sum(np.isnan(v)) / v.size)
    u_inf_ratio = float(np.sum(np.isinf(u)) / u.size)
    v_inf_ratio = float(np.sum(np.isinf(v)) / v.size)
    print('Estimated Flow nan ratio: u = {:.2f}, v = {:.2f}'
            .format(u_nan_ratio, v_nan_ratio))
    print('Estimated Flow inf ratio: u = {:.2f}, v = {:.2f}'
            .format(u_inf_ratio, v_inf_ratio))
    # Remove nan values from u and v
    u[np.isnan(u)] = 0
    v[np.isnan(v)] = 0

    return u, v


def lucas_kanade(u, v, dIx, dIy, dIt, param):
    # Set the neighborhood size for LK calulcations
    if param is None:
        ww = 5
    else:
        ww = param
    w = round((ww / 2) + 1e-9)  # Add to 1e-9 to account for odd integer ww

    for i in range(w, dIx.shape[0] - w):
        for j in range(w, dIx.shape[1] - w):
            # Extract the current window
            dIx_ww = dIx[(i - w):(i + w + 1), (j - w):(j + w + 1)]
            dIy_ww = dIy[(i - w):(i + w + 1), (j - w):(j + w + 1)]
            dIt_ww = dIt[(i - w):(i + w + 1), (j - w):(j + w + 1)]
            
            # Compute the flow vectors
            # HINT: Some useful functions:
            #   * ndarray.ravel()
            #   * np.stack()
            #   * np.linalg.lstsq()
            # ============================ EDIT THIS PART =====================
            A = np.transpose(np.vstack((np.ravel(dIx_ww), np.ravel(dIy_ww))))
            B = -np.ravel(dIt_ww)
            u[i,j], v[i,j] = np.linalg.lstsq(A,B, rcond=None)[0]

    # Compute and display the nan and inf ratios
    u_nan_ratio = float(np.sum(np.isnan(u)) / u.size)
    v_nan_ratio = float(np.sum(np.isnan(v)) / v.size)
    u_inf_ratio = float(np.sum(np.isinf(u)) / u.size)
    v_inf_ratio = float(np.sum(np.isinf(v)) / v.size)
    print('Estimated Flow nan ratio: u = {:.2f}, v = {:.2f}'
            .format(u_nan_ratio, v_nan_ratio))
    print('Estimated Flow inf ratio: u = {:.2f}, v = {:.2f}'
            .format(u_inf_ratio, v_inf_ratio))

    return u, v
