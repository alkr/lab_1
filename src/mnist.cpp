#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <functional>
#include <random>
#include <vector>
#include <map>
#include <string>
#include <algorithm>

using namespace std;

const float eps = 1e-6;
using Mat = vector<vector<float>>;

void mult (const Mat& mat, const vector<float>& vec, vector<float>& ans) {
	for (size_t i = 0; i < mat.size(); i++) {
		ans[i] = 0;
		for (size_t j = 0; j < mat[i].size(); j++) 
			ans[i] += mat[i][j] * vec[j];
	}
}

vector<function<float(float)>> funcs = {
	[](float x) { return max(x, 0.F); }, 
	[](float x) { return x; },
	[](float x) { return 1.F / (1 + exp(-x)); },
	[](float x) { return x; }
};	
vector<function<float(float)>> dfuncs = {
	[](float x) { return x > 0 ? 1 : 0; }, 
	[](float x) { return 1; },
	[](float x) { return x * (1 - x); },
	[](float x) { return x * (1 - x); }
};	

map<string, int> fInd = {
	{"relu", 0},
	{"to", 1},
	{"sigma", 2},
	{"softmax", 3}
};

struct Layer {
	int in, out, funcId;
	Mat mat, grad;
	function<float(float)> activate, df;
	Layer (int _in = 0, int _out = 0, const string& func = "to")
		: in(_in), out(_out), funcId(fInd[func]) {
			activate = funcs[funcId];
			df = dfuncs[funcId];
			mat.resize(out); grad.resize(out);			
			random_device rd;
			mt19937 gen(rd());
			float sigma = sqrt (12.F / (in + out));
			std::normal_distribution<double> distr_gauss(0, sigma);
			for (int i = 0; i < out; i++) {
				mat[i].resize(in);
				grad[i].resize(in);
				for (auto& x : mat[i]) x = distr_gauss(gen);
			}
		}

	void clear() {
		for (int i = 0; i < out; i++)
			for (int j = 0; j < in; j++)
				grad[i][j] = 0;
	}
};

struct Model {
	size_t n;
	vector<Layer> layers;
	vector<vector<float>> net;
	vector<vector<float>> out;
	vector<vector<float>> cur;
	vector<int> ans;
	
	float learningRate;
	function<float(float, float)> error = [](float x, float y) { return -y * log(x); };
	
	Model (const vector<int>& info, const vector<string>& func, float _learningRate) 
		: learningRate(_learningRate) {
			n = info.size();
			net.resize(n), out.resize(n), cur.resize(n);
			out[0].resize(info.front());
			cur[0].resize(info.front());
			for (size_t i = 1; i < info.size(); i++) {
				layers.push_back(Layer(info[i - 1], info[i], func[i]));
				net[i].resize(info[i]);
				out[i].resize(info[i]);
				cur[i].resize(info[i]);
			}
			ans.resize(out.back().size());
		}
 
	int check (int result) {
		for (int i = 0; i < (int) out.back().size(); i++) 
			ans[i] = i == result ? 1 : 0;
		float res = out.back()[0];
		int ind = 0;
		for (size_t i = 1; i < ans.size(); i++)
			if (out.back()[i] > res) {
			 	res = out.back()[i];
			 	ind = i;
			}
		return ind == result;
	}

	void go (const float* data) {
		for (size_t i = 0; i < out[0].size(); i++)
			out[0][i] = data[i];
		for (size_t i = 1; i < n; i++) {
			mult (layers[i - 1].mat, out[i - 1], net[i]);
			if (layers[i - 1].funcId == 3) {
				float denum = 0;
				for (size_t j = 0; j < out[i].size(); j++)
					denum += out[i][j] = exp(net[i][j]);
				for (auto& x : out[i]) x /= denum;
			} else {
				for (size_t j = 0; j < out[i].size(); j++)
					out[i][j] = layers[i - 1].activate(net[i][j]); 
			}
		}
	}

	void updGrads() {
		for (size_t i = n - 1; i; i--) {
			for (size_t j = 0; j < out[i].size(); j++) {
				cur[i][j] = (layers[i - 1].funcId == 2 || layers[i - 1].funcId == 3) 
								? layers[i - 1].df(out[i][j]) : layers[i - 1].df(net[i][j]);
				if (i == n - 1) cur[i][j] = out[i][j] - ans[j];
				else {
					float res = 0;
					for (size_t k = 0; k < out[i + 1].size(); k++)
						res += cur[i + 1][k] * layers[i].mat[k][j];
					cur[i][j] *= res;
				}
			}
			for (size_t k = 0; k < out[i].size(); k++)
				for (size_t j = 0; j < out[i - 1].size(); j++)
					layers[i - 1].grad[k][j] += cur[i][k] * out[i - 1][j];
		}
	}

	void backProp (float batch) {
		for (size_t i = n - 1; i; i--)
			for (size_t k = 0; k < out[i].size(); k++)
				for (size_t j = 0; j < out[i - 1].size(); j++)
					layers[i - 1].mat[k][j] -= learningRate * layers[i - 1].grad[k][j] / batch;
		for (size_t i = 0; i < n - 1; i++) layers[i].clear();
	}

	float train (const vector<vector<float>>& trainData, const vector<int>& results, int m, int batch) {
		vector<int> x(m);
		for (int i = 0; i < m; i++) x[i] = i;
		random_shuffle(x.begin(), x.end());
		float score = 0;
		for (int b = 0; b < m; b += batch) {
			int curSize = min (batch, m - b);
			for (int i = b; i < b + curSize; i++) {
				go(trainData[x[i]].data());
				score += check(results[x[i]]); 
				updGrads();
			}	
		   	backProp(batch);
		}
		return score / m; 
	}
};


const int N = 7e4;
const int S = 28;

int batch, epoch, close;
float learningRate; 

int main(int argc, char* argv[]) {
	srand(time(0));
	ios::sync_with_stdio();
	cin.tie(0); cout.tie(0);

	epoch = atoi(argv[1]);
	batch = atoi(argv[2]);
	close = atoi(argv[3]);
	learningRate = atof(argv[4]);

	vector<vector<float>> data(N);
	vector<int> ans(N);
		
	ifstream in("data.txt");
	for (int i = 0; i < N; i++) {
		in >> ans[i];
		data[i].resize(S * S);
		for (int j = 0; j < S * S; j++) {
			int cur;
			in >> cur;		
			data[i][j] = (float) cur / 255;
		}
	}
	in.close();	
	cerr << "START\n" << endl;
	
	auto timer = clock();

	vector<int> config = {784, close, 10};
	vector<string> funcs = {"to", "relu", "softmax"};
	Model model(config, funcs, learningRate);
	
	int n = 60000;
	for (int i = 0; i < epoch; i++) {
		cerr << "Epoch #" << i + 1 << " " << model.train(data, ans, n, batch) << "\n";
	}

	int res = 0;
	for (int i = n; i < N; i++) {
		model.go(data[i].data());
		res += model.check(ans[i]);
	}
	
	cerr << "Test accuracy: " << (float) res / (N - n) << "\n";
	cerr << "Time: " << (clock() - timer) / 1000 << "s\n";
}
