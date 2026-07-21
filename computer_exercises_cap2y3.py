"""
==============================================================
COMPUTER EXERCISES – Pattern Classification (Duda, Hart & Stork)
Capítulo 2: Bayesian Decision Theory
Capítulo 3: Maximum Likelihood and Bayesian Estimation
==============================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad
from scipy.special import erf
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Datos compartidos de la tabla (Caps. 2 y 3) ──────────────
# Capítulo 2
omega1_c2 = np.array([
    [-5.01, -8.12, -3.68],
    [-5.43, -3.48, -3.54],
    [ 1.08, -5.52,  1.66],
    [ 0.86, -3.78, -4.11],
    [-2.67,  0.63,  7.39],
    [ 4.94,  3.29,  2.08],
    [-2.51,  2.09, -2.59],
    [-2.25, -2.13, -6.94],
    [ 5.56,  2.86, -2.26],
    [ 1.03, -3.33,  4.33],
])
omega2_c2 = np.array([
    [-0.91, -0.18, -0.05],
    [ 1.30, -2.06, -3.53],
    [-7.75, -4.54, -0.95],
    [-5.47,  0.50,  3.92],
    [ 6.14,  5.72, -4.85],
    [ 3.60,  1.26,  4.36],
    [ 5.37, -4.63, -3.65],
    [ 7.18,  1.46, -6.66],
    [-7.39,  1.17,  6.30],
    [-7.50, -6.32, -0.31],
])
omega3_c2 = np.array([
    [ 5.35,  2.26,  8.13],
    [ 5.12,  3.22, -2.66],
    [-1.34, -5.31, -9.87],
    [ 4.48,  3.42,  5.19],
    [ 7.11,  2.39,  9.21],
    [ 7.17,  4.33, -0.98],
    [ 5.75,  3.97,  6.65],
    [ 0.77,  0.27,  2.41],
    [ 0.90, -0.43, -8.71],
    [ 3.52, -0.36,  6.43],
])

# Capítulo 3
omega1_c3 = np.array([
    [ 0.42, -0.087,  0.58],
    [-0.20, -3.300, -3.40],
    [ 1.30, -0.320,  1.70],
    [ 0.39,  0.710,  0.23],
    [-1.60, -5.300, -0.15],
    [-0.029, 0.890, -4.70],
    [-0.23,  1.900,  2.20],
    [ 0.27, -0.300, -0.87],
    [-1.90,  0.760, -2.10],
    [ 0.87, -1.000, -2.60],
])
omega2_c3 = np.array([
    [-0.400,  0.580,  0.0890],
    [-0.310,  0.270, -0.0400],
    [ 0.380,  0.055, -0.0350],
    [-0.150,  0.530,  0.0110],
    [-0.350,  0.470,  0.0340],
    [ 0.170,  0.690,  0.1000],
    [-0.011,  0.550, -0.1800],
    [-0.270,  0.610,  0.1200],
    [-0.065,  0.490,  0.0012],
    [-0.120,  0.054, -0.0630],
])
omega3_c3 = np.array([
    [ 0.83,  1.600, -0.014],
    [ 1.10,  1.600,  0.480],
    [-0.44, -0.410,  0.320],
    [ 0.047,-0.450,  1.400],
    [ 0.28,  0.350,  3.100],
    [-0.39, -0.480,  0.110],
    [ 0.34, -0.079,  0.140],
    [-0.30, -0.220,  2.200],
    [ 1.10,  1.200, -0.460],
    [ 0.18, -0.110, -0.490],
])

# ──────────────────────────────────────────────────────────────
# UTILIDADES COMUNES (Cap. 2, Ejercicio 1)
# ──────────────────────────────────────────────────────────────

def generate_normal_samples(mu, Sigma, n):
    """(1a) Genera n muestras de N(mu, Sigma) en d dimensiones."""
    return np.random.multivariate_normal(mu, Sigma, n)

def discriminant(x, mu, Sigma, prior):
    """
    (1b) Discriminante cuadrático Bayesiano (Eq. 47 del libro):
         g_i(x) = -½(x-µ)ᵀ Σ⁻¹(x-µ) - ½ ln|Σ| + ln P(ωᵢ)
    x puede ser un array (N, d) o vector (d,).
    """
    x = np.atleast_2d(x)
    d = mu.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    diff = x - mu
    mahal = np.einsum('ni,ij,nj->n', diff, Sigma_inv, diff)
    sign, logdet = np.linalg.slogdet(Sigma)
    return -0.5 * mahal - 0.5 * logdet + np.log(prior)

def euclidean_distance(a, b):
    """(1c) Distancia Euclidiana entre dos puntos."""
    return np.sqrt(np.sum((np.array(a) - np.array(b))**2))

def mahalanobis_distance(x, mu, Sigma):
    """(1d) Distancia de Mahalanobis entre x y µ dado Σ."""
    diff = np.array(x) - np.array(mu)
    Sigma_inv = np.linalg.inv(Sigma)
    return np.sqrt(diff @ Sigma_inv @ diff)

def mle_gaussian(data):
    """MLE de µ y Σ para datos Gaussianos. Retorna (mu, Sigma)."""
    mu = np.mean(data, axis=0)
    diff = data - mu
    Sigma = (diff.T @ diff) / len(data)   # divisor N (estimador MLE)
    return mu, Sigma

def classify_bayes(X, classes_data, priors):
    """
    Clasificador Bayesiano con MLE de parámetros.
    classes_data: lista de arrays de entrenamiento.
    priors: lista de probabilidades a priori.
    Retorna etiquetas predichas (0-based).
    """
    params = [mle_gaussian(d) for d in classes_data]
    X = np.atleast_2d(X)
    scores = np.stack([
        discriminant(X, mu, S, p)
        for (mu, S), p in zip(params, priors)
    ], axis=1)
    return np.argmax(scores, axis=1)

def empirical_error(y_true, y_pred):
    return np.mean(y_true != y_pred)

def bhattacharyya_bound(mu1, Sigma1, mu2, Sigma2, p1=0.5, p2=0.5):
    """
    Cota de Bhattacharyya para dos clases Gaussianas.
    P_e ≤ sqrt(P(ω1)*P(ω2)) * exp(-k_B)
    donde k_B = (1/8)(µ1-µ2)ᵀ Σ_avg⁻¹ (µ1-µ2)
                + (1/2) ln(|Σ_avg| / sqrt(|Σ1||Σ2|))
    con Σ_avg = (Σ1+Σ2)/2.
    """
    Sigma_avg = (Sigma1 + Sigma2) / 2
    diff = mu1 - mu2
    k1 = (1/8) * diff @ np.linalg.inv(Sigma_avg) @ diff
    _, ld_avg = np.linalg.slogdet(Sigma_avg)
    _, ld1    = np.linalg.slogdet(Sigma1)
    _, ld2    = np.linalg.slogdet(Sigma2)
    k2 = 0.5 * (ld_avg - 0.5*(ld1 + ld2))
    k_B = k1 + k2
    return np.sqrt(p1 * p2) * np.exp(-k_B)

# ═══════════════════════════════════════════════════════════════
# CAPÍTULO 2 – EJERCICIOS
# ═══════════════════════════════════════════════════════════════

def cap2_ejercicio2():
    """
    Ejercicio 2 – Clasificador Bayesiano para ω1 y ω2.
    (a) 1 feature (x1), (b) error empírico, (c) cota Bhattacharyya,
    (d) 2 features, (e) 3 features, (f) discusión.
    """
    print("\n" + "="*60)
    print("CAP. 2 – EJERCICIO 2: Clasificador ω1 vs ω2")
    print("="*60)

    clases = [omega1_c2, omega2_c2]
    priors = [0.5, 0.5]
    etiquetas = np.array([0]*10 + [1]*10)
    datos_completos = np.vstack(clases)

    for desc, feat_idx in [("(a) 1 feature (x1)", [0]),
                            ("(d) 2 features (x1,x2)", [0,1]),
                            ("(e) 3 features (x1,x2,x3)", [0,1,2])]:
        print(f"\n  {desc}")
        X_c1 = omega1_c2[:, feat_idx]
        X_c2 = omega2_c2[:, feat_idx]
        X    = np.vstack([X_c1, X_c2])

        mu1, S1 = mle_gaussian(X_c1)
        mu2, S2 = mle_gaussian(X_c2)

        y_pred = classify_bayes(X, [X_c1, X_c2], priors)
        err = empirical_error(etiquetas, y_pred)
        bhat = bhattacharyya_bound(mu1, S1, mu2, S2)

        print(f"    Error empírico de entrenamiento : {err*100:.1f}%")
        print(f"    Cota de Bhattacharyya           : {bhat:.4f}")

        if len(feat_idx) == 1:
            # (b) explícito
            print(f"  (b) Error empírico: {err*100:.1f}%")
            print(f"  (c) Cota Bhattacharyya: {bhat:.4f}")

    print("\n  (f) Discusión:")
    print("    Con más dimensiones el clasificador aprovecha más información,")
    print("    pero con solo 10 muestras por clase, la estimación de Σ puede")
    print("    volverse inestable. Con datos finitos, el error empírico en")
    print("    entrenamiento puede subir al añadir dimensiones (maldición de")
    print("    dimensionalidad en estimación), aunque el error de Bayes real")
    print("    nunca aumenta con más features relevantes.")


def cap2_ejercicio3():
    """Ejercicio 3 – Clasificador ω1 vs ω3."""
    print("\n" + "="*60)
    print("CAP. 2 – EJERCICIO 3: Clasificador ω1 vs ω3")
    print("="*60)
    priors = [0.5, 0.5]
    etiquetas = np.array([0]*10 + [1]*10)

    for desc, feat_idx in [("1 feature (x1)", [0]),
                            ("2 features (x1,x2)", [0,1]),
                            ("3 features", [0,1,2])]:
        X_c1 = omega1_c2[:, feat_idx]
        X_c2 = omega3_c2[:, feat_idx]
        X    = np.vstack([X_c1, X_c2])
        mu1, S1 = mle_gaussian(X_c1)
        mu2, S2 = mle_gaussian(X_c2)
        y_pred = classify_bayes(X, [X_c1, X_c2], priors)
        err  = empirical_error(etiquetas, y_pred)
        bhat = bhattacharyya_bound(mu1, S1, mu2, S2)
        print(f"  {desc:25s}  Error={err*100:.1f}%  Bhattacharyya={bhat:.4f}")


def cap2_ejercicio4():
    """Ejercicio 4 – Clasificador ω2 vs ω3."""
    print("\n" + "="*60)
    print("CAP. 2 – EJERCICIO 4: Clasificador ω2 vs ω3")
    print("="*60)
    priors = [0.5, 0.5]
    etiquetas = np.array([0]*10 + [1]*10)

    for desc, feat_idx in [("1 feature (x1)", [0]),
                            ("2 features (x1,x2)", [0,1]),
                            ("3 features", [0,1,2])]:
        X_c1 = omega2_c2[:, feat_idx]
        X_c2 = omega3_c2[:, feat_idx]
        X    = np.vstack([X_c1, X_c2])
        mu1, S1 = mle_gaussian(X_c1)
        mu2, S2 = mle_gaussian(X_c2)
        y_pred = classify_bayes(X, [X_c1, X_c2], priors)
        err  = empirical_error(etiquetas, y_pred)
        bhat = bhattacharyya_bound(mu1, S1, mu2, S2)
        print(f"  {desc:25s}  Error={err*100:.1f}%  Bhattacharyya={bhat:.4f}")


def cap2_ejercicio5():
    """
    Ejercicio 5 – Distancia de Mahalanobis y clasificación 3 clases.
    Puntos de prueba: (1,2,1), (5,3,2), (0,0,0), (1,0,0).
    """
    print("\n" + "="*60)
    print("CAP. 2 – EJERCICIO 5: Mahalanobis + 3 clases")
    print("="*60)

    test_pts = np.array([[1,2,1],[5,3,2],[0,0,0],[1,0,0]], dtype=float)
    clases   = [omega1_c2, omega2_c2, omega3_c2]
    nombres  = ['ω1','ω2','ω3']
    params   = [mle_gaussian(d) for d in clases]

    print("\n  (a) Distancias de Mahalanobis:")
    print(f"  {'Punto':15s}", "  ".join(f"{n:8s}" for n in nombres))
    for pt in test_pts:
        dists = [mahalanobis_distance(pt, mu, S) for mu, S in params]
        row = "  ".join(f"{d:8.3f}" for d in dists)
        print(f"  {str(pt):15s}  {row}")

    print("\n  (b) Clasificación con P(ωi)=1/3:")
    for pt in test_pts:
        scores = [discriminant(pt, mu, S, 1/3)[0] for mu, S in params]
        pred = nombres[np.argmax(scores)]
        print(f"    {str(pt):15s}  → {pred}")

    print("\n  (c) Clasificación con P(ω1)=0.8, P(ω2)=P(ω3)=0.1:")
    priors_c = [0.8, 0.1, 0.1]
    for pt in test_pts:
        scores = [discriminant(pt, mu, S, p)[0]
                  for (mu, S), p in zip(params, priors_c)]
        pred = nombres[np.argmax(scores)]
        print(f"    {str(pt):15s}  → {pred}")


def cap2_ejercicio6():
    """
    Ejercicio 6 – Teorema Central del Límite: promedio de uniformes
    converge a Gaussiana.
    (a) Genera n enteros de U(xl, xu).
    (b) Elige xl, xu, n aleatoriamente.
    (c-e) Histograma acumulado para 1e4, 1e5, 1e6 puntos.
    """
    print("\n" + "="*60)
    print("CAP. 2 – EJERCICIO 6: Teorema Central del Límite (TLC)")
    print("="*60)

    def generar_punto():
        """(b) Elige xl, xu y n aleatoriamente, devuelve la media."""
        xl = np.random.randint(-100, 99)
        xu = np.random.randint(xl+1, 101)
        n  = np.random.randint(1, 1001)
        muestra = np.random.randint(xl, xu+1, n)
        return np.mean(muestra)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, N in zip(axes, [10_000, 100_000, 1_000_000]):
        puntos = [generar_punto() for _ in range(N)]
        puntos = np.array(puntos)
        mu_emp = np.mean(puntos);  sigma_emp = np.std(puntos)
        ax.hist(puntos, bins=80, density=True, color='steelblue',
                alpha=0.7, label='Histograma')
        xs = np.linspace(puntos.min(), puntos.max(), 300)
        ax.plot(xs, norm.pdf(xs, mu_emp, sigma_emp), 'r-', lw=2,
                label=f'N(µ={mu_emp:.1f}, σ={sigma_emp:.1f})')
        ax.set_title(f'N={N:,}')
        ax.set_xlabel('Media de muestra')
        ax.legend(fontsize=7)
    plt.suptitle('Ejercicio 6 – TLC: acumulación de medias de uniformes')
    plt.tight_layout()
    plt.savefig('c2_ej6_TLC.png', dpi=100)
    plt.close()

    print("  Con más puntos (1e4 → 1e6), el histograma se aproxima cada vez")
    print("  más a una Gaussiana (TLC). Gráfico guardado: c2_ej6_TLC.png")


def cap2_ejercicio7():
    """
    Ejercicio 7 – Error empírico vs cota de Bhattacharyya.
    p(x|ω1)~N([1,0], I),  p(x|ω2)~N([-1,0], I),  P(ω1)=P(ω2)=1/2.
    La frontera de Bayes es el eje x2=0 con x1=0 (plano x1=0).
    """
    print("\n" + "="*60)
    print("CAP. 2 – EJERCICIO 7: Error empírico vs Bhattacharyya")
    print("="*60)

    mu1 = np.array([1.0, 0.0]);  mu2 = np.array([-1.0, 0.0])
    S   = np.eye(2)
    bhat = bhattacharyya_bound(mu1, S, mu2, S)
    print(f"  (b) Frontera de Bayes: x1 = 0  (plano perpendicular a eje x1)")
    print(f"  Cota de Bhattacharyya: {bhat:.4f}")

    ns      = list(range(100, 1100, 100))
    errores = []
    for n in ns:
        X1 = np.random.multivariate_normal(mu1, S, n//2)
        X2 = np.random.multivariate_normal(mu2, S, n//2)
        X  = np.vstack([X1, X2])
        y  = np.array([0]*(n//2) + [1]*(n//2))
        y_pred = classify_bayes(X, [X1, X2], [0.5, 0.5])
        errores.append(empirical_error(y, y_pred))
        print(f"  n={n:5d}: error empírico = {errores[-1]*100:.1f}%")

    # True error: norm.cdf(-1) pues d/2=1
    true_err = norm.cdf(-1.0)
    print(f"\n  Error de Bayes verdadero: {true_err:.4f}")
    print(f"  Cota Bhattacharyya      : {bhat:.4f}")
    print("  Nota: el error empírico puede superar la cota Bhattacharyya")
    print("  con muestras pequeñas, pero nunca lo hace la cota del error verdadero.")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ns, [e*100 for e in errores], 'bo-', label='Error empírico')
    ax.axhline(true_err*100,  color='g', linestyle='--', label=f'Error Bayes {true_err*100:.1f}%')
    ax.axhline(bhat*100, color='r', linestyle=':',  label=f'Cota Bhattacharyya {bhat*100:.1f}%')
    ax.set_xlabel('n (muestras totales)');  ax.set_ylabel('Error (%)')
    ax.set_title('Ejercicio 7 – Error empírico vs Bhattacharyya')
    ax.legend();  plt.tight_layout()
    plt.savefig('c2_ej7_error_vs_bhat.png', dpi=100)
    plt.close()
    print("  Gráfico guardado: c2_ej7_error_vs_bhat.png")


def cap2_ejercicio8():
    """
    Ejercicio 8 – p(x|ω1)~N(-0.5,1), p(x|ω2)~N(+0.5,1), P=0.5.
    (a) Cota Bhattacharyya.
    (b-c) Error verdadero con erf y numéricamente.
    (d-e) Error empírico vs n y comparación.
    """
    print("\n" + "="*60)
    print("CAP. 2 – EJERCICIO 8: Error verdadero vs empírico 1D")
    print("="*60)

    mu1, mu2, sig = -0.5, 0.5, 1.0

    # (a) Bhattacharyya 1D
    k_B = (mu1 - mu2)**2 / (8 * sig**2)
    bhat = 0.5 * np.exp(-k_B)
    print(f"  (a) Cota Bhattacharyya = {bhat:.4f}")

    # (b-c) Error verdadero
    # Umbral óptimo: x* = (mu1+mu2)/2 = 0  (priors iguales, sigmas iguales)
    # P_e = P(ω1)*P(x>0|ω1) + P(ω2)*P(x<0|ω2)
    #      = 0.5*[1-Φ((0-mu1)/sig)] + 0.5*Φ((0-mu2)/sig)
    # Con mu1=-0.5, mu2=+0.5: P(x>0|ω1) = P(z>0.5) y P(x<0|ω2) = P(z<-0.5)
    true_err_analytic = norm.cdf((mu1 - 0) / sig)   # por simetría
    # En términos de erf: Φ(z) = 0.5*(1+erf(z/sqrt(2)))
    # P_e = 0.5*(1 - erf(0.5/sqrt(2)))
    true_err_erf = 0.5 * (1 - erf(0.5 / np.sqrt(2)))
    true_err_num, _ = quad(
        lambda x: 0.5*norm.pdf(x, mu1, sig), 0, np.inf
    )
    true_err_num += quad(
        lambda x: 0.5*norm.pdf(x, mu2, sig), -np.inf, 0
    )[0]
    print(f"  (b) Error verdadero (erf)       = {true_err_erf:.4f}")
    print(f"  (c) Error verdadero (numérico)  = {true_err_num:.4f}")

    # (d-e) Error empírico como función de n
    ns      = [10, 50, 100, 200, 500, 1000]
    errores = []
    for n in ns:
        X1 = np.random.normal(mu1, sig, n)
        X2 = np.random.normal(mu2, sig, n)
        # Reestimar µ y σ para cada conjunto
        mu1_hat = np.mean(X1);  mu2_hat = np.mean(X2)
        # Umbral Bayes re-estimado (priors iguales, σ iguales)
        thr = (mu1_hat + mu2_hat) / 2
        X_all = np.concatenate([X1, X2])
        y_all = np.array([0]*n + [1]*n)
        y_pred = (X_all >= thr).astype(int)
        errores.append(empirical_error(y_all, y_pred))

    print(f"\n  (d-e) Error empírico por n:")
    for n, e in zip(ns, errores):
        print(f"    n={n:5d}  error={e*100:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(ns, [e*100 for e in errores], 'bo-', label='Error empírico')
    ax.axhline(true_err_erf*100, color='g', linestyle='--',
               label=f'Error verdadero {true_err_erf*100:.2f}%')
    ax.axhline(bhat*100, color='r', linestyle=':',
               label=f'Bhattacharyya {bhat*100:.2f}%')
    ax.set_xlabel('n');  ax.set_ylabel('Error (%)')
    ax.set_title('Ejercicio 8 – Convergencia del error empírico')
    ax.legend();  plt.tight_layout()
    plt.savefig('c2_ej8_convergencia_error.png', dpi=100)
    plt.close()
    print("  Gráfico guardado: c2_ej8_convergencia_error.png")


def cap2_ejercicio9():
    """
    Ejercicio 9 – Variaciones del ejercicio 8.
    (a) N(-0.5,2) vs N(+0.5,2), P=2/3, 1/3
    (b) N(-0.5,2) vs N(+0.5,2), P=1/2
    (c) N(-0.5,3) vs N(+0.5,1), P=1/2
    """
    print("\n" + "="*60)
    print("CAP. 2 – EJERCICIO 9: Variaciones del Ej. 8")
    print("="*60)

    casos = [
        ("(a) N(-0.5,√2) vs N(+0.5,√2), P=2/3,1/3",
         -0.5, 0.5, np.sqrt(2), np.sqrt(2), 2/3, 1/3),
        ("(b) N(-0.5,√2) vs N(+0.5,√2), P=1/2,1/2",
         -0.5, 0.5, np.sqrt(2), np.sqrt(2), 0.5, 0.5),
        ("(c) N(-0.5,√3) vs N(+0.5,1), P=1/2,1/2",
         -0.5, 0.5, np.sqrt(3), 1.0, 0.5, 0.5),
    ]

    for desc, mu1, mu2, sig1, sig2, p1, p2 in casos:
        print(f"\n  {desc}")
        # Bhattacharyya 1D general
        S_avg = (sig1**2 + sig2**2) / 2
        k1 = (mu1 - mu2)**2 / (8 * S_avg)
        k2 = 0.5 * np.log(S_avg / (sig1 * sig2))
        bhat = np.sqrt(p1 * p2) * np.exp(-(k1 + k2))
        print(f"    Cota Bhattacharyya = {bhat:.4f}")

        # Error numérico (integramos la región de decisión incorrecta)
        # Umbral: resolver p1*N(x|mu1,sig1) = p2*N(x|mu2,sig2)
        # Numéricamente buscamos en [-5, 5]
        xs = np.linspace(-10, 10, 10000)
        g1 = norm.pdf(xs, mu1, sig1) * p1
        g2 = norm.pdf(xs, mu2, sig2) * p2
        # Decide clase 1 donde g1>=g2
        decide1 = g1 >= g2
        err_num = (np.trapezoid(g2[decide1],  xs[decide1]) +
                   np.trapezoid(g1[~decide1], xs[~decide1]))
        print(f"    Error verdadero (numérico) ≈ {err_num:.4f}")

        n = 500
        X1 = np.random.normal(mu1, sig1, n)
        X2 = np.random.normal(mu2, sig2, n)
        mu1h, mu2h = np.mean(X1), np.mean(X2)
        sig1h, sig2h = np.std(X1), np.std(X2)
        g1_h = norm.pdf(xs, mu1h, sig1h+1e-9) * p1
        g2_h = norm.pdf(xs, mu2h, sig2h+1e-9) * p2
        thr_idx = np.argmin(np.abs(g1_h - g2_h))
        thr = xs[thr_idx]
        X_all = np.concatenate([X1, X2])
        y_all = np.array([0]*n + [1]*n)
        # Regla: clase 2 si x >= thr (asumiendo mu2 > mu1)
        if mu2 > mu1:
            y_pred = (X_all >= thr).astype(int)
        else:
            y_pred = (X_all < thr).astype(int)
        print(f"    Error empírico (n={n})     = {empirical_error(y_all, y_pred)*100:.1f}%")


# ═══════════════════════════════════════════════════════════════
# CAPÍTULO 3 – EJERCICIOS
# ═══════════════════════════════════════════════════════════════

def cap3_ejercicio1():
    """
    Ejercicio 1 – MLE de µ y Σ en diferentes dimensiones.
    (a) 1D por feature, (b) 2D combinaciones, (c) 3D completo,
    (d) modelo separable en ω2, (e-f) comparación.
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 1: MLE Gaussiano multidimensional")
    print("="*60)

    # (a) 1D: cada feature de ω1
    print("\n  (a) MLE 1D – cada feature de ω1:")
    for i, fname in enumerate(['x1','x2','x3']):
        xi = omega1_c3[:, i]
        mu_hat    = np.mean(xi)
        sigma_hat = np.std(xi, ddof=0)
        print(f"    {fname}: µ̂={mu_hat:7.4f}  σ̂²={sigma_hat**2:7.4f}")

    # (b) 2D: pares de features de ω1
    print("\n  (b) MLE 2D – pares de features de ω1:")
    pairs = [(0,1,'x1,x2'), (0,2,'x1,x3'), (1,2,'x2,x3')]
    for i, j, name in pairs:
        data2d = omega1_c3[:, [i,j]]
        mu2, S2 = mle_gaussian(data2d)
        print(f"    {name}: µ̂={np.round(mu2,4)}  Σ̂=\n      {np.round(S2,4)}")

    # (c) 3D completo de ω1
    print("\n  (c) MLE 3D – ω1 completo:")
    mu3, S3 = mle_gaussian(omega1_c3)
    print(f"    µ̂ = {np.round(mu3, 4)}")
    print(f"    Σ̂ =\n{np.round(S3, 4)}")

    # (d) Modelo separable (Σ diagonal) en ω2
    print("\n  (d) Modelo separable en ω2 (Σ = diag):")
    mu_d, S_d = mle_gaussian(omega2_c3)
    diag_sigma = np.diag(np.diag(S_d))
    print(f"    µ̂ = {np.round(mu_d, 4)}")
    print(f"    Σ̂_diag = {np.round(np.diag(diag_sigma), 4)}")

    # (e-f) Comparación
    print("\n  (e) La media µ̂_i es idéntica en todos los casos:")
    print("      MLE de µ en cualquier subconjunto de features es la media marginal.")
    print("  (f) La varianza σ̂²_i es la misma si estimamos 1D o diagonal de Σ̂,")
    print("      ya que MLE minimiza cada término independientemente.")


def cap3_ejercicio2():
    """
    Ejercicio 2 – Densidad triangular y estimación Bayesiana.
    T(µ, δ): densidad triangular. Aplicada a x2 de ω2.
    Usamos una cuadrícula para calcular p(x|D) ∝ Π p(xk|θ) p(θ).
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 2: Densidad triangular – estimación Bayesiana")
    print("="*60)

    datos = omega2_c3[:, 1]   # x2 de ω2

    def triangular_pdf(x, mu, delta):
        """p(x|µ,δ) = (δ - |x-µ|)/δ² si |x-µ|<δ, 0 si no."""
        val = (delta - np.abs(x - mu)) / delta**2
        val = np.where(np.abs(x - mu) < delta, val, 0.0)
        return val

    # Cuadrícula de parámetros θ = (µ, δ)
    mu_grid    = np.linspace(-1.0, 2.0, 150)
    delta_grid = np.linspace(0.05, 3.0, 150)
    MU, DELTA  = np.meshgrid(mu_grid, delta_grid)

    # Prior uniforme sobre la cuadrícula
    log_posterior = np.zeros_like(MU)
    for xk in datos:
        p_vals = triangular_pdf(xk, MU, DELTA)
        log_posterior += np.log(np.maximum(p_vals, 1e-300))

    # Normalizar
    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()

    # p(x|D) = ∫ p(x|θ) p(θ|D) dθ  (marginalizamos sobre θ)
    x_test = np.linspace(-2, 4, 200)
    px_D   = np.zeros_like(x_test)
    for k, xv in enumerate(x_test):
        px_D[k] = np.sum(triangular_pdf(xv, MU, DELTA) * posterior)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].contourf(mu_grid, delta_grid, posterior, levels=20, cmap='viridis')
    axes[0].set_xlabel('µ');  axes[0].set_ylabel('δ')
    axes[0].set_title('Posterior p(µ,δ|D)')
    axes[1].plot(x_test, px_D, 'b-', lw=2)
    axes[1].scatter(dados:=datos, np.zeros_like(datos)-0.005, c='red',
                    marker='|', s=100, label='Datos x₂ de ω₂')
    axes[1].set_xlabel('x');  axes[1].set_ylabel('p(x|D)')
    axes[1].set_title('Densidad posterior predictiva p(x|D)')
    axes[1].legend()
    plt.suptitle('Ejercicio 2 – Densidad triangular Bayesiana')
    plt.tight_layout()
    plt.savefig('c3_ej2_triangular.png', dpi=100)
    plt.close()
    print("  Gráfico guardado: c3_ej2_triangular.png")

    mu_MAP  = mu_grid[np.unravel_index(posterior.argmax(), posterior.shape)[1]]
    del_MAP = delta_grid[np.unravel_index(posterior.argmax(), posterior.shape)[0]]
    print(f"  Estimación MAP: µ̂={mu_MAP:.3f}, δ̂={del_MAP:.3f}")


def cap3_ejercicio3():
    """
    Ejercicio 3 – Estimación Bayesiana de la media de una Gaussiana 1D.
    Prior: p(µ) ~ N(µ0, σ0).  Posterior: N(µn, σn).
    (a) p(x|D) como función de n.
    (b) x2 de ω3; µ0=-1; dogmatismo σ²/σ0²: 0.1, 1, 10, 100.
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 3: Estimación Bayesiana de la media")
    print("="*60)

    datos = omega3_c3[:, 1]   # x2 de ω3
    n     = len(datos)
    sigma = np.std(datos, ddof=0)   # σ estimada (asumida conocida)
    x_bar = np.mean(datos)
    mu0   = -1.0

    print(f"  Datos x2 de ω3: n={n}, x̄={x_bar:.4f}, σ̂={sigma:.4f}")

    x_plot = np.linspace(-5, 4, 300)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Evolución de p(x|D) al aumentar n
    ax = axes[0]
    sigma0 = sigma   # dogmatismo = σ²/σ0² = 1
    for n_sub in [1, 3, 5, 10]:
        D  = datos[:n_sub]
        x_bar_sub = np.mean(D)
        # Actualización Bayesiana:
        # µn = (σ0² * Σxk + σ² * µ0) / (n*σ0² + σ²)
        # σn² = σ0²σ² / (n*σ0² + σ²)
        sigma0_sq = sigma0**2;  sigma_sq = sigma**2
        mu_n    = (sigma0_sq * n_sub * x_bar_sub + sigma_sq * mu0) \
                  / (n_sub * sigma0_sq + sigma_sq)
        sigma_n = np.sqrt(sigma0_sq * sigma_sq / (n_sub * sigma0_sq + sigma_sq))
        # p(x|D) = N(x | µn, sqrt(σ² + σn²))
        sigma_pred = np.sqrt(sigma_sq + sigma_n**2)
        ax.plot(x_plot, norm.pdf(x_plot, mu_n, sigma_pred),
                label=f'n={n_sub}')
    ax.set_title('(a) p(x|D) con creciente n (dogmatismo=1)')
    ax.set_xlabel('x');  ax.set_ylabel('p(x|D)')
    ax.legend();  ax.axvline(x_bar, color='k', linestyle=':', label='x̄')

    # (b) Distintos dogmatismos con n=10
    ax = axes[1]
    dogmatismos = [0.1, 1.0, 10.0, 100.0]
    for dog in dogmatismos:
        sigma0_sq = sigma**2 / dog
        sigma_sq  = sigma**2
        mu_n    = (sigma0_sq * n * x_bar + sigma_sq * mu0) \
                  / (n * sigma0_sq + sigma_sq)
        sigma_n = np.sqrt(sigma0_sq * sigma_sq / (n * sigma0_sq + sigma_sq))
        sigma_pred = np.sqrt(sigma_sq + sigma_n**2)
        ax.plot(x_plot, norm.pdf(x_plot, mu_n, sigma_pred),
                label=f'σ²/σ₀²={dog}')
        print(f"  Dogmatismo={dog:5}: µ_n={mu_n:.4f}, σ_n={sigma_n:.4f}")
    ax.set_title('(b) p(x|D) para distintos dogmatismos (n=10)')
    ax.set_xlabel('x');  ax.set_ylabel('p(x|D)')
    ax.legend()
    plt.suptitle('Ejercicio 3 – Estimación Bayesiana de µ Gaussiano')
    plt.tight_layout()
    plt.savefig('c3_ej3_bayes_media.png', dpi=100)
    plt.close()
    print("  Gráfico guardado: c3_ej3_bayes_media.png")


def cap3_ejercicio4():
    """
    Ejercicio 4 – Estimación Bayesiana recursiva para densidad uniforme 2D.
    Datos: x1-x2 de ω1. Prior: caja [-6,6]x[-6,6].
    p(x|θ) ~ U(xl, xu); actualización recursiva del posterior.
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 4: Estimación Bayesiana recursiva – Uniforme 2D")
    print("="*60)

    datos2D = omega1_c3[:, 0:2]   # x1, x2 de ω1 (10 puntos)

    # Cuadrícula de parámetros: (xl1, xu1) y (xl2, xu2)
    # Para reducir costo computacional usamos una cuadrícula moderada
    bounds = np.linspace(-8, 8, 50)
    # Prior: xl ∈ [-6,-6], xu ∈ [+6,+6] → solo mantenemos xl<xu
    xl1_g = np.linspace(-6, 0, 20)
    xu1_g = np.linspace(0, 6, 20)
    xl2_g = np.linspace(-6, 0, 20)
    xu2_g = np.linspace(0, 6, 20)

    # Representamos el posterior como un vector 4D (xl1,xu1,xl2,xu2)
    # Inicialmente uniforme donde xl<xu
    XL1, XU1, XL2, XU2 = np.meshgrid(xl1_g, xu1_g, xl2_g, xu2_g,
                                       indexing='ij')
    valid = (XL1 < XU1) & (XL2 < XU2)
    log_post = np.where(valid, 0.0, -np.inf)

    n_obs = 10
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for k in range(2, n_obs+1):
        D = datos2D[:k]
        # Actualización: log p += log U(xk | xl, xu)
        for xk in D:
            x1k, x2k = xk
            in_box = ((XL1 <= x1k) & (x1k <= XU1) &
                      (XL2 <= x2k) & (x2k <= XU2))
            # log p(xk|θ) = -log(|xu1-xl1|*|xu2-xl2|) si dentro, -inf si no
            area = (XU1-XL1) * (XU2-XL2)
            log_lv = np.where(in_box & (area > 0),
                               -np.log(np.maximum(area, 1e-12)), -np.inf)
            log_post = log_post + log_lv
            log_post = np.where(valid, log_post, -np.inf)
        # Normalizar (evitar overflow)
        lp_finite = log_post[np.isfinite(log_post)]
        if len(lp_finite) > 0:
            log_post -= lp_finite.max()

        # Marginalizar sobre xl2, xu2 para visualizar en x1
        post = np.exp(log_post)
        post_x1 = post.sum(axis=(2,3))   # marginal sobre (xl1, xu1)

        ax = axes.flatten()[k-2]
        ax.imshow(post_x1, origin='lower',
                  extent=[xu1_g[0], xu1_g[-1], xl1_g[0], xl1_g[-1]],
                  aspect='auto', cmap='hot')
        ax.scatter(D[:,0], np.full(k, xl1_g[0]+0.3), c='cyan', s=10)
        ax.set_title(f'n={k}', fontsize=8)
        ax.set_xlabel('xu1', fontsize=7);  ax.set_ylabel('xl1', fontsize=7)

    plt.suptitle('Ejercicio 4 – Posterior p(xl1,xu1|D) marginalizado')
    plt.tight_layout()
    plt.savefig('c3_ej4_uniforme_recursivo.png', dpi=100)
    plt.close()
    print("  Gráfico guardado: c3_ej4_uniforme_recursivo.png")
    print("  La región de alta probabilidad se concentra alrededor")
    print("  de los extremos observados de los datos (MLE: xl̂=min, xû=max).")


def cap3_ejercicio5():
    """
    Ejercicio 5 – Estadísticas suficientes para la familia exponencial.
    Datos: x3 de ω3.
    Familias: Gaussiana, Rayleigh, Maxwell.
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 5: Estadísticas suficientes – familia exponencial")
    print("="*60)

    datos = omega3_c3[:, 2]   # x3 de ω3
    n     = len(datos)
    print(f"  Datos x3 de ω3: {np.round(datos,4)}")

    # ── Gaussiana: p(x|µ,σ²) = exp(-(x-µ)²/(2σ²)) / sqrt(2πσ²)
    # Estadísticas suficientes: T1=Σxi, T2=Σxi²
    T_gauss_1 = np.sum(datos)
    T_gauss_2 = np.sum(datos**2)
    mu_hat_g  = T_gauss_1 / n
    var_hat_g = T_gauss_2/n - mu_hat_g**2
    print(f"\n  Gaussiana:")
    print(f"    T1=Σxi={T_gauss_1:.4f},  T2=Σxi²={T_gauss_2:.4f}")
    print(f"    µ̂ = T1/n = {mu_hat_g:.4f}")
    print(f"    σ̂² = T2/n - µ̂² = {var_hat_g:.4f}")

    # ── Rayleigh: p(x|σ) = (x/σ²) exp(-x²/(2σ²)),  x>0
    # Estadística suficiente: T = Σxi²
    datos_pos = np.abs(datos)   # usamos |x| para evitar x<0
    T_rayleigh = np.sum(datos_pos**2)
    sigma_hat_ray = np.sqrt(T_rayleigh / (2*n))
    print(f"\n  Rayleigh (usando |x3|):")
    print(f"    T=Σxi²={T_rayleigh:.4f}")
    print(f"    σ̂ = sqrt(T/(2n)) = {sigma_hat_ray:.4f}")

    # ── Maxwell: p(x|a) = sqrt(2/π) (x²/a³) exp(-x²/(2a²)),  x>0
    # Estadística suficiente: T = Σxi²
    T_maxwell = T_rayleigh
    a_hat = np.sqrt(T_maxwell / (3*n))
    print(f"\n  Maxwell (usando |x3|):")
    print(f"    T=Σxi²={T_maxwell:.4f}")
    print(f"    â = sqrt(T/(3n)) = {a_hat:.4f}")

    # Visualización: densidades estimadas
    xv = np.linspace(0.01, 4, 200)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(xv, norm.pdf(xv, mu_hat_g, np.sqrt(var_hat_g)), 'b')
    axes[0].set_title(f'Gaussiana\nµ̂={mu_hat_g:.2f}, σ̂={np.sqrt(var_hat_g):.2f}')
    # Rayleigh pdf
    ray_pdf = (xv/sigma_hat_ray**2)*np.exp(-xv**2/(2*sigma_hat_ray**2))
    axes[1].plot(xv, ray_pdf, 'g')
    axes[1].set_title(f'Rayleigh\nσ̂={sigma_hat_ray:.2f}')
    # Maxwell pdf
    max_pdf = np.sqrt(2/np.pi)*(xv**2/a_hat**3)*np.exp(-xv**2/(2*a_hat**2))
    axes[2].plot(xv, max_pdf, 'r')
    axes[2].set_title(f'Maxwell\nâ={a_hat:.2f}')
    for ax in axes:
        ax.hist(datos_pos, bins=5, density=True, alpha=0.4, color='gray')
        ax.set_xlabel('x')
    plt.suptitle('Ejercicio 5 – Densidades MLE familia exponencial')
    plt.tight_layout()
    plt.savefig('c3_ej5_exponencial.png', dpi=100)
    plt.close()
    print("  Gráfico guardado: c3_ej5_exponencial.png")


def cap3_ejercicio6():
    """
    Ejercicio 6 – Error en distintas dimensiones: ω1 vs ω2.
    (a) 3D, (b) tres subespacios 2D, (c) tres subespacios 1D, (d-e) discusión.
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 6: Error en distintas dimensiones (ω1 vs ω2)")
    print("="*60)

    def error_gaussiano_numerico(mu1, S1, mu2, S2, p1=0.5, p2=0.5, n_mc=100_000):
        """Error Bayesiano via Monte Carlo."""
        X1 = np.random.multivariate_normal(mu1, S1, n_mc//2)
        X2 = np.random.multivariate_normal(mu2, S2, n_mc//2)
        X  = np.vstack([X1, X2])
        y  = np.array([0]*(n_mc//2) + [1]*(n_mc//2))
        g1 = multivariate_normal.logpdf(X, mu1, S1) + np.log(p1)
        g2 = multivariate_normal.logpdf(X, mu2, S2) + np.log(p2)
        y_pred = (g2 > g1).astype(int)
        return empirical_error(y, y_pred)

    X1 = omega1_c3;  X2 = omega2_c3
    mu1, S1 = mle_gaussian(X1);  mu2, S2 = mle_gaussian(X2)

    print("\n  (a) 3D completo:")
    err_3d = error_gaussiano_numerico(mu1, S1, mu2, S2)
    print(f"    Error ≈ {err_3d*100:.1f}%")

    print("\n  (b) Subespacios 2D (eliminando una feature):")
    for feat_elim, idx in [('x1=0',[1,2]), ('x2=0',[0,2]), ('x3=0',[0,1])]:
        m1, s1 = mle_gaussian(X1[:,idx]);  m2, s2 = mle_gaussian(X2[:,idx])
        err = error_gaussiano_numerico(m1, s1, m2, s2)
        print(f"    {feat_elim}: Error ≈ {err*100:.1f}%")

    print("\n  (c) Subespacios 1D (cada eje):")
    for i, fname in enumerate(['x1','x2','x3']):
        d1 = X1[:,i:i+1];  d2 = X2[:,i:i+1]
        m1, s1 = mle_gaussian(d1);  m2, s2 = mle_gaussian(d2)
        err = error_gaussiano_numerico(m1, s1, m2, s2)
        print(f"    {fname}: Error ≈ {err*100:.1f}%")

    print("\n  (d) Discusión sobre el orden de errores:")
    print("    Generalmente el error disminuye con más dimensiones si las")
    print("    distribuciones difieren en esas direcciones. Con muestras")
    print("    pequeñas (n=10), la estimación de Σ puede ser ruidosa.")
    print("  (e) El error de Bayes en el espacio proyectado nunca puede ser")
    print("    menor que en el espacio completo (teorema de suficiencia).")


def cap3_ejercicio7():
    """Ejercicio 7 – Como ejercicio 6 pero para ω1 vs ω3."""
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 7: Error en distintas dimensiones (ω1 vs ω3)")
    print("="*60)

    def err_mc(X1, X2, idx, n_mc=100_000):
        d1 = X1[:, idx];  d2 = X2[:, idx]
        m1, s1 = mle_gaussian(d1);  m2, s2 = mle_gaussian(d2)
        Xs = np.vstack([
            np.random.multivariate_normal(m1, s1, n_mc//2),
            np.random.multivariate_normal(m2, s2, n_mc//2)
        ])
        y  = np.array([0]*(n_mc//2) + [1]*(n_mc//2))
        g1 = multivariate_normal.logpdf(Xs, m1, s1)
        g2 = multivariate_normal.logpdf(Xs, m2, s2)
        return empirical_error(y, (g2>g1).astype(int))

    X1 = omega1_c3;  X3 = omega3_c3
    print(f"  3D:  Error ≈ {err_mc(X1,X3,[0,1,2])*100:.1f}%")
    for lbl, idx in [('x1=0',[1,2]),('x2=0',[0,2]),('x3=0',[0,1])]:
        print(f"  2D ({lbl}): Error ≈ {err_mc(X1,X3,idx)*100:.1f}%")
    for i, n in enumerate(['x1','x2','x3']):
        print(f"  1D ({n}):  Error ≈ {err_mc(X1,X3,[i])*100:.1f}%")


def cap3_ejercicio8():
    """
    Ejercicio 8 – Clasificación con shrinkage de covarianzas.
    (a) Genera datos, (b) estima µ y Σ, (c) shrinkage según α,
    (d) error de entrenamiento vs α, (e) error de test vs α.
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 8: Shrinkage de covarianzas")
    print("="*60)

    mu1_t = np.array([0., 0., 0.]);  S1_t = np.diag([3., 5., 2.])
    mu2_t = np.array([1., 5., -3.]); S2_t = np.array([[1,0,0],[0,4,1],[0,1,6]],
                                                       dtype=float)
    mu3_t = np.array([0., 0., 0.]);  S3_t = 10*np.eye(3)

    # (a) Generar datos de entrenamiento y test
    n_train, n_test = 20, 50
    X1_tr = np.random.multivariate_normal(mu1_t, S1_t, n_train)
    X2_tr = np.random.multivariate_normal(mu2_t, S2_t, n_train)
    X3_tr = np.random.multivariate_normal(mu3_t, S3_t, n_train)
    X1_te = np.random.multivariate_normal(mu1_t, S1_t, n_test)
    X2_te = np.random.multivariate_normal(mu2_t, S2_t, n_test)
    X3_te = np.random.multivariate_normal(mu3_t, S3_t, n_test)

    # (b) Estimar parámetros
    mu_hat = [np.mean(d, axis=0) for d in [X1_tr, X2_tr, X3_tr]]
    S_hat  = [np.cov(d, rowvar=False, ddof=0) for d in [X1_tr, X2_tr, X3_tr]]
    S_pooled = sum(S_hat) / 3   # covarianza común pooled

    def shrink(S_i, S_common, alpha):
        """Σ_α = (1-α)*Σ_i + α*Σ_común  (Eq. 76 del libro)."""
        return (1-alpha)*S_i + alpha*S_common

    def bayes_error(X_list, y, mu_list, S_list, priors):
        X_all = np.vstack(X_list)
        scores = np.stack([
            discriminant(X_all, m, S, p)
            for m, S, p in zip(mu_list, S_list, priors)
        ], axis=1)
        y_pred = np.argmax(scores, axis=1)
        return empirical_error(y, y_pred)

    alphas = np.linspace(0, 1, 41)
    y_tr = np.array([0]*n_train + [1]*n_train + [2]*n_train)
    y_te = np.array([0]*n_test  + [1]*n_test  + [2]*n_test)
    X_tr_all = [X1_tr, X2_tr, X3_tr]
    X_te_all = [X1_te, X2_te, X3_te]
    priors = [1/3]*3

    errs_tr, errs_te = [], []
    for alpha in alphas:
        S_shrunk = [shrink(S, S_pooled, alpha) for S in S_hat]
        errs_tr.append(bayes_error(X_tr_all, y_tr, mu_hat, S_shrunk, priors))
        errs_te.append(bayes_error(X_te_all, y_te, mu_hat, S_shrunk, priors))

    # (d-e) Gráfica
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(alphas, [e*100 for e in errs_tr], 'b-o', ms=3, label='Error entrenim.')
    ax.plot(alphas, [e*100 for e in errs_te], 'r-s', ms=3, label='Error test')
    ax.set_xlabel('α (shrinkage)');  ax.set_ylabel('Error (%)')
    ax.set_title('Ejercicio 8 – Shrinkage de covarianzas')
    ax.legend();  plt.tight_layout()
    plt.savefig('c3_ej8_shrinkage.png', dpi=100)
    plt.close()

    best_alpha = alphas[np.argmin(errs_te)]
    print(f"  α óptimo (min error test) ≈ {best_alpha:.2f}")
    print(f"  Error train en α=0 (sin shrinkage): {errs_tr[0]*100:.1f}%")
    print(f"  Error test  en α=0                : {errs_te[0]*100:.1f}%")
    print(f"  Error test  en α óptimo           : {min(errs_te)*100:.1f}%")
    print("  Gráfico guardado: c3_ej8_shrinkage.png")


def cap3_ejercicio9():
    """
    Ejercicio 9 – Algoritmo EM con datos faltantes.
    ω1 3D: x3 faltante para puntos pares.
    Inicio: µ0=0, Σ0=I.
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 9: EM con datos faltantes (ω1, x3)")
    print("="*60)

    datos = omega1_c3.copy()   # (10,3)
    # x3 faltante para índices 1,3,5,7,9 (0-based: pares del libro = impares aquí)
    missing_idx = [1, 3, 5, 7, 9]   # filas con x3 faltante

    # Datos completos (sin missing)
    mu_full, S_full = mle_gaussian(datos)
    print(f"  MLE sin datos faltantes: µ̂={np.round(mu_full,4)}")
    print(f"  Σ̂=\n{np.round(S_full,4)}")

    # EM
    mu_em = np.zeros(3)
    S_em  = np.eye(3)
    n_iter = 50

    for it in range(n_iter):
        # E-step: imputar x3 faltante mediante esperanza condicional Gaussiana
        datos_imp = datos.copy()
        for i in missing_idx:
            # p(x3 | x1, x2, θ) es Gaussiana condicional
            # Particionamos: obs=[x1,x2], miss=[x3]
            mu_obs  = mu_em[:2];  mu_miss  = mu_em[2:3]
            S_oo = S_em[:2,:2];   S_mo = S_em[2:3,:2]
            S_oo_inv = np.linalg.inv(S_oo + 1e-9*np.eye(2))
            x_obs = datos[i, :2]
            # E[x3|x1,x2] = mu_miss + S_mo @ S_oo^-1 @ (x_obs - mu_obs)
            x3_imp = mu_miss + S_mo @ S_oo_inv @ (x_obs - mu_obs)
            datos_imp[i, 2] = x3_imp[0]

        # M-step: re-estimar µ y Σ con datos completados
        mu_em, S_em = mle_gaussian(datos_imp)

    print(f"\n  EM (µ estimada): {np.round(mu_em,4)}")
    print(f"  EM (Σ estimada):\n{np.round(S_em,4)}")
    print("\n  Comparación µ:")
    print(f"    Sin missing: {np.round(mu_full,4)}")
    print(f"    Con EM     : {np.round(mu_em,4)}")
    print("  → EM recupera una estimación cercana a la original incluso")
    print("    con el 50% de los valores x3 faltantes.")


def cap3_ejercicio10():
    """
    Ejercicio 10 – EM para distribución uniforme 3D con datos faltantes.
    ω2 3D: x3 faltante para puntos pares.
    Inicio: xl=(-2,-2,-2), xu=(+2,+2,+2).
    MLE de uniforme: xl̂=min, xû=max.
    """
    print("\n" + "="*60)
    print("CAP. 3 – EJERCICIO 10: EM datos faltantes – Uniforme 3D (ω2)")
    print("="*60)

    datos = omega2_c3.copy()
    missing_idx = [1, 3, 5, 7, 9]

    # Estimación completa
    xl_full = datos.min(axis=0)
    xu_full = datos.max(axis=0)
    print(f"  MLE sin faltantes: xl̂={np.round(xl_full,4)}  xû={np.round(xu_full,4)}")

    # EM
    xl_em = np.array([-2., -2., -2.])
    xu_em = np.array([ 2.,  2.,  2.])
    n_iter = 50

    for it in range(n_iter):
        # E-step: imputar x3 faltante como la media de U(xl3, xu3)
        datos_imp = datos.copy()
        for i in missing_idx:
            # E[x3 | x1,x2, xl,xu] = (xl3+xu3)/2 si U es separable
            datos_imp[i, 2] = (xl_em[2] + xu_em[2]) / 2

        # M-step: re-estimar xl, xu con datos completados
        xl_em = datos_imp.min(axis=0)
        xu_em = datos_imp.max(axis=0)

    print(f"  EM final          : xl̂={np.round(xl_em,4)}  xû={np.round(xu_em,4)}")
    print("\n  Comparación:")
    print(f"    Sin faltantes  xl3={xl_full[2]:.4f}, xu3={xu_full[2]:.4f}")
    print(f"    Con EM (50% faltante) xl3={xl_em[2]:.4f}, xu3={xu_em[2]:.4f}")
    print("  → EM tiende a subestimar el rango de x3 al imputar con la media,")
    print("    aunque converge a un estimado razonable para las features observadas.")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" COMPUTER EXERCISES – CAPÍTULO 2")
    print("="*60)
    cap2_ejercicio2()
    cap2_ejercicio3()
    cap2_ejercicio4()
    cap2_ejercicio5()
    cap2_ejercicio6()
    cap2_ejercicio7()
    cap2_ejercicio8()
    cap2_ejercicio9()

    print("\n\n" + "="*60)
    print(" COMPUTER EXERCISES – CAPÍTULO 3")
    print("="*60)
    cap3_ejercicio1()
    cap3_ejercicio2()
    cap3_ejercicio3()
    cap3_ejercicio4()
    cap3_ejercicio5()
    cap3_ejercicio6()
    cap3_ejercicio7()
    cap3_ejercicio8()
    cap3_ejercicio9()
    cap3_ejercicio10()

    print("\n" + "="*60)
    print("Todos los ejercicios completados.")
    print("="*60)
