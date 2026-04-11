import numpy as np

def generate_grid(N_side, output_file):
    N_total = N_side * N_side
    D_eff = 0.15 / np.sqrt(N_total)

    lines = []
    lines.append(f"* Mem4ristor v3 - {N_side}x{N_side} Coupled Network")
    lines.append(f"* Network: {N_side}x{N_side} = {N_total} units with 4-neighbor periodic coupling")
    lines.append("")
    lines.append(f".title Mem4ristor v3 Coupled Network ({N_side}x{N_side})")
    lines.append("")
    lines.append("* ── Global Parameters ──")
    lines.append(".param a=0.7")
    lines.append(".param b=0.8")
    lines.append(".param eps=0.08")
    lines.append(".param alpha_cog=0.15")
    lines.append(".param eps_u=0.02")
    lines.append(".param sigma_base=0.05")
    lines.append(f".param D_eff={D_eff:.6f}")
    lines.append(".param dt=0.05")
    lines.append("")
    lines.append("* ── Unit Time Constant (RC = 1) ──")
    lines.append(".param R_int=1")
    lines.append(".param C_int=1")
    lines.append("")

    # Integrators
    lines.append("* ── State Variables (RC Integrators) ──")
    for i in range(N_total):
        lines.append(f"* Unit {i}")
        lines.append(f"R_v{i} v{i}_node v{i}_int {{R_int}}")
        # Initialize some randomly
        init_v = np.random.uniform(-1.0, 1.0)
        lines.append(f"C_v{i} v{i}_int 0 {{C_int}} IC={init_v:.2f}")
        lines.append(f"R_w{i} w{i}_node w{i}_int {{R_int}}")
        lines.append(f"C_w{i} w{i}_int 0 {{C_int}} IC=0.0")
        lines.append(f"R_u{i} u{i}_node u{i}_int {{R_int}}")
        lines.append(f"C_u{i} u{i}_int 0 {{C_int}} IC=0.05")
        lines.append("")

    # Dynamics
    lines.append("* ── Unit Dynamics ──")
    for i in range(N_total):
        # find periodic neighbors
        row = i // N_side
        col = i % N_side
        up = ((row - 1) % N_side) * N_side + col
        down = ((row + 1) % N_side) * N_side + col
        left = row * N_side + ((col - 1) % N_side)
        right = row * N_side + ((col + 1) % N_side)
        
        neighbors = [up, down, left, right]
        laplacian_terms = " + ".join([f"(V(v{n}_int) - V(v{i}_int))" for n in neighbors])
        
        lines.append(f"* Unit {i}: dv/dt")
        lines.append(f"B_dv{i} v{i}_node 0 V = V(v{i}_int) - pow(V(v{i}_int),3)/5 - V(w{i}_int) + D_eff * (tanh(3.14159*(0.5 - V(u{i}_int))) + 0.05) * ({laplacian_terms}) - alpha_cog*tanh(V(v{i}_int))")
        lines.append(f"B_dw{i} w{i}_node 0 V = eps * (V(v{i}_int) + a - b * V(w{i}_int))")
        lines.append(f"B_du{i} u{i}_node 0 V = eps_u * (sigma_base - V(u{i}_int))")
        lines.append("")

    # Simulation and control
    lines.append("* ── Simulation ──")
    lines.append(".tran 0.1 50 uic")
    lines.append("")
    lines.append("* ── Analysis ──")
    lines.append(".control")
    lines.append("run")
    lines.append("echo \"========================================\"")
    lines.append(f"echo \"  Mem4ristor v3 - {N_side}x{N_side} Network\"")
    lines.append("echo \"========================================\"")
    lines.append("")
    lines.append(f"let v0_final = v(v0_int)[length(v(v0_int))-1]")
    lines.append(f"let v_half = v(v{N_total//2}_int)[length(v(v{N_total//2}_int))-1]")
    lines.append("")
    lines.append("echo \"Final states (sample):\"")
    lines.append(f"echo \"  v[0] = $&v0_final\"")
    lines.append(f"echo \"  v[{N_total//2}] = $&v_half\"")
    lines.append("quit")
    lines.append(".endc")
    lines.append("")
    lines.append(".end")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Generated {output_file} successfully.")

if __name__ == '__main__':
    generate_grid(5, 'mem4ristor_coupled_5x5.cir')
    generate_grid(10, 'mem4ristor_coupled_10x10.cir')
