
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
using namespace std;

// == Настройки ==
static const char* PIVOT = "partial"; // partial | row | full
static const size_t N1 = 6;         // размер для пункта 1
static const size_t N2 = 8;         // размер для пункта 2 (Hilbert)

// == Типы и счётчик ==
using Matrix = vector<vector<double>>;
using Vector = vector<double>;
struct Ops { long long add = 0, mul = 0, divv = 0, cmp = 0; void reset() { add = mul = divv = cmp = 0; } };

// == Печать (fixed) ==
void printVec(const Vector& v, const string& name) {
	cout << name << " = [ ";
	for (size_t i = 0; i < v.size(); ++i) cout << v[i] << (i + 1 == v.size() ? " ]\n" : " ");
}
void printMat(const Matrix& A, const string& name) {
	cout << name << ":\n";
	for (const auto& row : A) {
		for (double x : row) cout << setw(12) << x << ' ';
		cout << '\n';
	}
	cout << '\n';
}

// == Нормы ==
double norm1_vec(const Vector& x) { double s = 0; for (double v : x) s += fabs(v); return s; }
double norm1_mat(const Matrix& A) {
	size_t n = A.size(), m = A[0].size(); double r = 0;
	for (size_t j = 0; j < m; ++j) { double s = 0; for (size_t i = 0; i < n; ++i) s += fabs(A[i][j]); r = max(r, s); }
	return r;
}
double normInf_mat(const Matrix& A) {
	double r = 0; for (const auto& row : A) { double s = 0; for (double v : row) s += fabs(v); r = max(r, s); } return r;
}

// == Базовые операции ==
Vector matvec(const Matrix& A, const Vector& x, Ops* op = nullptr) {
	size_t n = A.size(), m = A[0].size(); Vector y(n, 0.0);
	for (size_t i = 0; i < n; ++i) {
		double s = 0.0;
		for (size_t j = 0; j < m; ++j) {
			s += A[i][j] * x[j];
			if (op) { op->mul++; op->add++; }
		}
		y[i] = s;
	}
	return y;
}

// == Генерации ==
Matrix random_SPD(size_t n, uint32_t seed = 42, double alpha = 1.0) {
	mt19937 rng(seed); uniform_real_distribution<double> U(-1.0, 1.0);
	Matrix M(n, vector<double>(n));
	for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) M[i][j] = U(rng);
	Matrix A(n, vector<double>(n, 0.0));
	for (size_t i = 0; i < n; ++i)
		for (size_t k = 0; k < n; ++k)
			for (size_t j = 0; j < n; ++j)
				A[i][j] += M[k][i] * M[k][j];
	for (size_t i = 0; i < n; ++i) A[i][i] += alpha;
	return A;
}
Matrix hilbert(size_t n) {
	Matrix A(n, vector<double>(n));
	for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) A[i][j] = 1.0 / double(i + j + 1);
	return A;
}

// == Перестановка столбцов ==
struct Perm {
	vector<size_t> p; explicit Perm(size_t n = 0) { 
		p.resize(n); for (size_t i = 0; i < n; ++i) p[i] = i; 
	}
	void swap_pos(size_t a, size_t b) { 
		swap(p[a], p[b]);
	}
	void apply_inverse(Vector& x) const { 
		Vector y(x.size()); for (size_t j = 0; j < x.size(); ++j) y[j] = x[p[j]]; x.swap(y); 
	}
};

// == Гаусс (прямой ход) ==
struct FR { Matrix U; Vector b; double det_sign; Perm colperm; bool singular; };
FR forward_elimination(Matrix A, Vector b, const string& mode, Ops& op) {
	size_t n = A.size(); double det_sign = 1.0; Perm cp(n); bool sing = false;
	for (size_t k = 0; k < n && !sing; ++k) {
		size_t piv_i = k, piv_j = k; double best = 0.0;
		if (mode == "partial") {
			for (size_t i = k; i < n; ++i) { 
				op.cmp++; double v = fabs(A[i][k]); if (v > best) { best = v; piv_i = i;
				} 
			}
		}
		else if (mode == "row") {
			for (size_t j = k; j < n; ++j) { 
				op.cmp++; double v = fabs(A[k][j]); 
				if (v > best) { 
					best = v; piv_j = j; 
				} 
			}
		}
		else if (mode == "full") {
			for (size_t i = k; i < n; ++i) for (size_t j = k; j < n; ++j) {
				op.cmp++; double v = fabs(A[i][j]); 
				if (v > best) { 
					best = v; piv_i = i; piv_j = j; 
				}
			}
		}
		else throw runtime_error("pivot must be partial|row|full");

		if (piv_i != k) { swap(A[piv_i], A[k]); swap(b[piv_i], b[k]); det_sign = -det_sign; }
		if ((mode == "row" || mode == "full") && piv_j != k) { 
			for (size_t i = 0; i < n; ++i) swap(A[i][piv_j], A[i][k]); cp.swap_pos(k, piv_j); 
		}

		double piv = A[k][k];
		if (fabs(piv) <= numeric_limits<double>::epsilon()) { sing = true; break; }

		for (size_t i = k + 1; i < n; ++i) {
			double m = A[i][k] / piv; op.divv++; A[i][k] = 0.0;
			for (size_t j = k + 1; j < n; ++j) { A[i][j] -= m * A[k][j]; op.mul++; op.add++; }
			b[i] -= m * b[k]; op.mul++; op.add++;
		}
	}
	return { A,b,det_sign,cp,sing };
}

// == Обратный ход ==
Vector back_subst(const Matrix& U, const Vector& b, Ops& op) {
	size_t n = U.size(); Vector x(n, 0.0);
	for (int i = (int)n - 1; i >= 0; --i) {
		double s = b[i];
		for (size_t j = i + 1; j < n; ++j) {
			s -= U[i][j] * x[j]; op.mul++; op.add++; 
		}
		x[i] = s / U[i][i]; op.divv++;
	}
	return x;
}

// == Обращение (для cond) ==
Matrix invert_via_gauss(const Matrix& Ain, const string& mode, Ops& op, bool* ok = nullptr) {
	size_t n = Ain.size(); Matrix A = Ain, Inv(n, vector<double>(n, 0.0)); for (size_t i = 0; i < n; ++i) Inv[i][i] = 1.0;
	Perm cp(n); bool sing = false;
	for (size_t k = 0; k < n && !sing; ++k) {
		size_t piv_i = k, piv_j = k; double best = 0.0;
		if (mode == "partial") { 
			for (size_t i = k; i < n; ++i) { 
				op.cmp++; double v = fabs(A[i][k]); if (v > best) { best = v; piv_i = i; 
				} 
			} 
		}
		else if (mode == "row") { 
			for (size_t j = k; j < n; ++j) {
				op.cmp++; double v = fabs(A[k][j]); if (v > best) { best = v; piv_j = j; 
				} 
			} 
		}
		else { 
			for (size_t i = k; i < n; ++i) for (size_t j = k; j < n; ++j) { 
				op.cmp++; double v = fabs(A[i][j]); if (v > best) { best = v; piv_i = i; piv_j = j; 
				} 
			} 
		}
		if (piv_i != k) { 
			swap(A[piv_i], A[k]); swap(Inv[piv_i], Inv[k]); 
		}
		if ((mode == "row" || mode == "full") && piv_j != k) { 
			for (size_t i = 0; i < n; ++i) { swap(A[i][piv_j], A[i][k]); swap(Inv[i][piv_j], Inv[i][k]); 
			} 
			cp.swap_pos(k, piv_j); 
		}
		double piv = A[k][k]; if (fabs(piv) <= numeric_limits<double>::epsilon()) { 
			sing = true; break; 
		}
		double invp = 1.0 / piv; op.divv++;
		for (size_t j = 0; j < n; ++j) { 
			A[k][j] *= invp; Inv[k][j] *= invp; op.mul += 2;
		}
		for (size_t i = 0; i < n; ++i) {
			if (i == k) continue; double m = A[i][k]; if (m == 0.0) continue;
			for (size_t j = 0; j < n; ++j) { 
				A[i][j] -= m * A[k][j]; Inv[i][j] -= m * Inv[k][j]; op.mul += 2; op.add += 2; 
			}
		}
	}
	if (ok) *ok = !sing; if (sing) return Inv;
	if (mode == "row" || mode == "full") {
		Matrix F(n, vector<double>(n, 0.0));
		for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) F[i][j] = Inv[i][cp.p[j]];
		return F;
	}
	return Inv;
}

// == Теоретический счёт ==
struct Theory { long long add = 0, mul = 0, divv = 0; };
Theory theoretical_counts(size_t n) {
	long long S1 = (long long)n*(n - 1) / 2;           // ∑ (n-k) = ∑ (i-1)
	long long S2 = (long long)(n - 1)*n*(2 * n - 1) / 6;   // ∑ (n-k)^2
	Theory t; t.mul = S2 + S1; t.add = S2 + S1; t.divv = S1 + n; return t;
}

// == Один прогон ==
void run_and_show(const string& caption, const Matrix& A0, const Vector& b0, const string& pivot) {
	cout << "\n=== " << caption << " | pivot=" << pivot << " ===\n\n";

	Matrix A = A0; Vector b = b0; Ops op;
	auto fr = forward_elimination(A, b, pivot, op);
	if (fr.singular) { cout << "Система численно вырождена.\n"; return; }

	Vector x = back_subst(fr.U, fr.b, op);
	if (pivot == "row" || pivot == "full")
		fr.colperm.apply_inverse(x); // вернуть перестановку столбцов

	printMat(A0, "A");
	printVec(b0, "b");
	printMat(fr.U, "U");
	printVec(x, "x");

	double detA = fr.det_sign; for (size_t i = 0; i < fr.U.size(); ++i) detA *= fr.U[i][i];
	cout << "det(A) = " << detA << "\n";

	Ops opi; bool ok = false; Matrix Ainv = invert_via_gauss(A0, pivot, opi, &ok);
	if (ok) {
		cout << "cond_1(A)   = " << norm1_mat(A0)*norm1_mat(Ainv) << "\n";
		cout << "cond_inf(A) = " << normInf_mat(A0)*normInf_mat(Ainv) << "\n";
	}
	else {
		cout << "cond(A): обращение не удалось.\n";
	}

	Vector r = matvec(A0, x, nullptr); for (size_t i = 0; i < r.size(); ++i) r[i] -= b0[i];
	cout << "||r||_1 = " << norm1_vec(r) << "\n";

	auto th = theoretical_counts(A0.size());
	cout << "\n[Операции]\n";
	cout << "add=" << op.add << ", mul=" << op.mul << ", div=" << op.divv << ", cmp=" << op.cmp << "\n";
	cout << "Теоретические: add=" << th.add << ", mul=" << th.mul << ", div=" << th.divv << "\n";
}

// == main ==
int main() {
	setlocale(LC_ALL, "Russian");
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	
	cout.setf(ios::fmtflags(0), ios::floatfield);
	cout << fixed << setprecision(6);

	// Пункт 1 — случайная SPD
	{
		Matrix A = random_SPD(N1, 42u, 1.0);
		Vector x_true(N1, 1.0);         // можно заменить на случайный, но для печати удобнее 1
		Ops tmp; Vector b = matvec(A, x_true, &tmp);
		run_and_show("Пункт 1. Случайная SPD система", A, b, PIVOT);
	}

	// Пункт 2 — Hilbert, x* = 1, b = A*1
	{
		Matrix A = hilbert(N2);
		Vector x_true(N2, 1.0);
		Ops tmp; Vector b = matvec(A, x_true, &tmp);
		run_and_show("Пункт 2. Матрица Гильберта (x*=1, b=A*1)", A, b, PIVOT);
	}

	return 0;
}