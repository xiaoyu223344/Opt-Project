import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# SVM 类实现，采用 SMO 算法优化
class SVM:
    def __init__(self, C=1.0, tol=1e-3, max_iter=10):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.w = None
        self.b = 0
        self.alphas = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alphas = np.zeros(n_samples)
        self.w = np.zeros(n_features)
        self.b = 0

        it = 0
        with tqdm(total=self.max_iter, desc="Training Progress") as pbar:
            while it < self.max_iter:
                alpha_prev = np.copy(self.alphas)
                for i in range(n_samples):
                    xi, yi = X[i], y[i]
                    gi = np.dot(self.w, xi) + self.b
                    ei = gi - yi

                    if (yi * gi < 1 and self.alphas[i] < self.C) or (yi * gi > 1 and self.alphas[i] > 0):
                        j = np.random.choice([x for x in range(n_samples) if x != i])
                        xj, yj = X[j], y[j]
                        gj = np.dot(self.w, xj) + self.b
                        ej = gj - yj

                        alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                        if yi != yj:
                            L = max(0, self.alphas[j] - self.alphas[i])
                            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                        else:
                            L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                            H = min(self.C, self.alphas[j] + self.alphas[i])

                        if L == H:
                            continue

                        eta = 2 * np.dot(xi, xj) - np.dot(xi, xi) - np.dot(xj, xj)
                        if eta >= 0:
                            continue

                        self.alphas[j] -= yj * (ei - ej) / eta
                        self.alphas[j] = np.clip(self.alphas[j], L, H)

                        if abs(self.alphas[j] - alpha_j_old) < self.tol:
                            continue

                        self.alphas[i] += yi * yj * (alpha_j_old - self.alphas[j])

                        b1 = self.b - ei - yi * (self.alphas[i] - alpha_i_old) * np.dot(xi, xi) - yj * (self.alphas[j] - alpha_j_old) * np.dot(xi, xj)
                        b2 = self.b - ej - yi * (self.alphas[i] - alpha_i_old) * np.dot(xi, xj) - yj * (self.alphas[j] - alpha_j_old) * np.dot(xj, xj)

                        if 0 < self.alphas[i] < self.C:
                            self.b = b1
                        elif 0 < self.alphas[j] < self.C:
                            self.b = b2
                        else:
                            self.b = (b1 + b2) / 2

                diff = np.linalg.norm(self.alphas - alpha_prev)
                if diff < self.tol:
                    break
                it += 1
                pbar.update(1)

        self.w = np.dot((self.alphas * y).T, X)

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# 读取数据文件并预处理
def load_data(file_path, pairs):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[0])
            if label in pairs:
                features = np.zeros(28 * 28)
                for feature in parts[1:]:
                    index, value = feature.split(':')
                    features[int(index)] = float(value)
                data.append(features)
                labels.append(1 if label == pairs[0] else -1)
    return np.array(data), np.array(labels)

# 训练和测试
def train_and_test(train_file, test_file, pairs):
    # 加载训练数据
    X_train, y_train = load_data(train_file, pairs)

    # 初始化并训练 SVM
    svm = SVM(C=1.0)
    svm.fit(X_train, y_train)

    # 加载测试数据
    X_test, y_test = load_data(test_file, pairs)
    predictions = svm.predict(X_test)

    # 计算准确率
    accuracy = np.mean(predictions == y_test)
    print(f"组合 {pairs}: 测试准确率 = {accuracy:.2f}")

    # 绘制部分结果
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(predictions)), predictions, c=y_test, cmap='bwr', alpha=0.6)
    plt.title(f"SVM 预测结果 (组合 {pairs})")
    plt.xlabel("样本")
    plt.ylabel("预测值")
    plt.show()

# 主函数
if __name__ == "__main__":
    train_file = 'F:\\Code\\Opt Project\\Data\\mnist.training.scale.txt'
    test_file = 'F:\\Code\\Opt Project\\Data\\mnist.testing.scale.txt'
    pairs_list = [(4, 9), (4, 6), (0, 1), (2, 7)]

    for pairs in pairs_list:
        train_and_test(train_file, test_file, pairs)