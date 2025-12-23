#APF discriminator
import os
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



class ExpertCurveClassifier:
    def __init__(self, expert_save_id: int = 726):
        cache_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         f'expert_{expert_save_id}_cache.npz')
        )
        data = np.load(cache_path, allow_pickle=True)
        self.coef = data['coef']
        self.intercept = data['intercept']
        self.bandwidth = float(data['bandwidth'])
        self.X_expert = data['X']  # 用于绘图

    # 手动预测 y
    def _poly_feat(self, x):
        # 手写 4 次多项式特征，避免重新创建 PolynomialFeatures
        return np.array([1, x, x**2, x**3, x**4])

    def f_curve(self, x):
        return np.polyval(self.coef[::-1], x) + self.intercept

    def classify(self,speed, distance):
        y_pred = self.f_curve(speed)
        delta = abs(distance - y_pred)

        if delta <= self.bandwidth:
            return 0
        elif distance < y_pred - self.bandwidth:
            return 1 #危险的信号
        else:
            return -1 #安全的信号

    def plot_old(self, new_sample=None, save_path=None):
        import matplotlib.pyplot as plt
        x_line = np.linspace(self.X_expert[:, 0].min(),
                             self.X_expert[:, 0].max(), 500)
        y_line = self.f_curve(x_line)

        plt.figure(figsize=(6, 4))
        plt.scatter(self.X_expert[:, 0], self.X_expert[:, 1],
                    s=15, alpha=0.3, label='Expert Data')
        plt.plot(x_line, y_line, color='crimson', lw=2, ls='--', label='Fitted Curve')
        plt.fill_between(x_line,
                         y_line - self.bandwidth,
                         y_line + self.bandwidth,
                         color='crimson', alpha=0.1, label='Curve Band')
        plt.xlim(0, 12.05)   # x 轴范围
        plt.ylim(0, 25)   # y 轴范围
        
        if new_sample:
            plt.scatter(*new_sample, color='blue', marker='x', s=60)
        plt.xlabel('AV_speed'); plt.ylabel('AV_distance')
        plt.legend(); plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300); plt.close()
        else:
            plt.show()

    def plot(self, new_sample=None, save_path=None):
        x_vals = self.X_expert[:, 0]
        y_vals = self.X_expert[:, 1]
        y_curve = self.f_curve(x_vals)

        # 计算到曲线的带符号距离
        delta = y_vals - y_curve

        # 三色分组
        mask_lower = delta < -self.bandwidth          # 下方
        mask_upper = delta >  self.bandwidth          # 上方
        mask_mid   = ~mask_lower & ~mask_upper        # 带宽内

        plt.figure(figsize=(8, 6))

        # 带宽内
        plt.scatter(x_vals[mask_mid], y_vals[mask_mid], edgecolors="#0DC3F1", linewidths=0.2,   
                s=15, c="#8ED8EA",  alpha=0.6, label='expert data(r_adv=0)')

        # 曲线上方
        plt.scatter(x_vals[mask_upper], y_vals[mask_upper], edgecolors="#0DE961", linewidths=0.2, 
                s=15, c="#7DF59D",    alpha=0.6, label='above band(r_adv=1)')

        # 曲线下方
        plt.scatter(x_vals[mask_lower], y_vals[mask_lower], edgecolors="#FB0869", linewidths=0.2, 
                  s=15, c="#F783A8",   alpha=0.6, label='below band(r_adv=-1)')

        # 曲线 & 带宽
        x_line = np.linspace(0, 12, 500)
        y_line = self.f_curve(x_line)
        plt.plot(x_line, y_line, color='black', lw=2, ls='--', label='Fitted Curve')
        plt.fill_between(x_line,
                     y_line - self.bandwidth,
                     y_line + self.bandwidth,
                     color='crimson', alpha=0.1, label='Curve Band')

        if new_sample:
            plt.scatter(*new_sample, color='orange', marker='x', s=60)

        plt.xlim(0, 12)
        plt.ylim(0, 25)
        plt.xlabel('AV:speed')
        plt.ylabel('AV:distance from ahead OV')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(
                    save_path,
                    dpi=600,              # 分辨率（可提高到 600 或更高）
                    facecolor='white',    # 防止透明背景导致锯齿
                    edgecolor='none',     # 无边框
                    transparent=False,    # 不透明背景
                    format='png',         # 建议使用 PNG（无损）
                    pil_kwargs={'compress_level': 0}  # PNG 无压缩
                    )
            plt.close()
        else:
            plt.show()


# clf = ExpertCurveClassifier(expert_save_id=726)
# reward = clf.classify(3.2, 25.7)
# reward = clf.classify(3.3, 25.7)
# print(reward)
# clf.plot(save_path='figure3.png')

