# 中文全流程复现指南

## 1. 首次安装

建议使用 64 位 Windows 10/11、至少 16 GB 内存。完整 1,280 个模型的实验建议使用 32 GB 内存和 NVIDIA GPU；快速验证不需要 GPU。

本项目在 Windows 上推荐使用 **Anaconda Prompt** 和 Conda 环境。不需要单独安装 Python，不需要运行 `python -m venv`、`Set-ExecutionPolicy` 或 `Activate.ps1`。Conda 会在独立环境内安装论文所用的 Python 3.13.5。

1. 安装 Anaconda 或 Miniconda。
2. 从 Windows 开始菜单打开 **Anaconda Prompt**，不要使用普通 PowerShell。
3. 下载或克隆 GitHub 仓库。在文件资源管理器中打开仓库文件夹，单击地址栏复制实际路径，然后在 Anaconda Prompt 输入 `cd /d`、一个空格，再粘贴路径。例如：

```bat
cd /d "C:\path\to\MICO-PMInet-reproducibility"
```

4. 如果是第一次在这台电脑上安装，推荐直接运行自动安装和验收脚本：

```bat
setup_conda_windows.bat
```

该脚本会自动创建 `mico-pminet` 环境、安装固定版本依赖、安装本项目、运行测试并执行 `python run.py verify`。安装成功时最后显示 `[SUCCESS]`。

也可以不用脚本，逐行执行完全相同的首次创建流程：

```bat
conda env create -f environment.yml
conda activate mico-pminet
python -m pip install --no-deps -e .
```

如果 `mico-pminet` 环境已经存在，则这已不属于“首次创建”，不要再次运行创建脚本，改为执行：

```bat
conda activate mico-pminet
conda env update -n mico-pminet -f environment.yml
python -m pip install --no-deps -e .
```

若创建时下载较慢或出现 HTTP 超时，先激活已创建的环境，再使用清华 PyPI 镜像安装依赖：

```bat
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 600 --retries 10
python -m pip install --no-deps -e .
```

每次重新打开 Anaconda Prompt 后，只需重新进入仓库并激活环境：

```bat
cd /d "C:\path\to\MICO-PMInet-reproducibility"
conda activate mico-pminet
```

当命令行开头出现 `(mico-pminet)` 时，表示环境激活成功。可执行 `python --version`，应显示 Python 3.13.5。

macOS/Linux 用户在仓库根目录执行：

```bash
bash setup_conda_unix.sh
```

仓库路径可以位于任意位置；两个安装脚本都会自动定位到自身所在的仓库根目录。支持 64 位 Windows、macOS 和 Linux。由于 PyTorch 和科学计算包依赖操作系统架构，建议使用仍受 Conda/PyTorch 支持的常见 64 位电脑。

如果电脑没有独立显卡，上述安装会使用 CPU。若要使用 CUDA，请根据 PyTorch 官方安装器为 `torch==2.9.1` 安装与你显卡驱动匹配的版本，然后再安装其余依赖。

## 2. 先做快速验收

```bat
python run.py verify
python -m pytest -q
```

通过标准：数据行数、467 个波数特征、64/16/8 个动物划分、8 个器官完整性、最终模型结构、论文关键指标与 SHAP 器官排序均显示通过。

## 3. 从原始光谱重做预处理

```bat
python run.py preprocess
```

该命令按作者确认的实际实验顺序执行：读取6,336条原始光谱，对每条技术重复光谱分别截取1800-900 cm⁻¹并进行SNV和airPLS，然后将每份样本的9条预处理光谱平均成代表光谱，最后按动物划分训练/测试/未见时间点数据。结果写入 `results/processed/`，不会覆盖仓库自带的基准数据。修订稿方法部分也应明确写成这一顺序。

随后核对新旧处理结果：

```bat
python run.py verify --processed-dir results/processed
```

## 4. 复现单器官 PLS

```bat
python run.py pls --processed-dir results/processed
```

输出为 `results/pls/pls_metrics.csv` 和 `results/pls/pls_predictions.csv`。论文重点参考值包括脑和肝Test-R²约0.87、Test-RMSE约2.9 h，玻璃体Test-R²约0.62、Test-RMSE约4.94 h。

## 5. 先试跑深度模型

```bat
python run.py train --profile smoke --processed-dir results/processed
```

此模式只训练一个小模型、一个 beta、一个折，目的是检查安装和显存/内存，不用于论文结果。

## 6. 复现最终 OSB 模型

```bat
python run.py select
```

该命令从 16 个模型 × 8 个 beta 的 128 行十折汇总中，复算论文最终模型 `OSB-WMHA-AWA-MORM` 的 beta 选择，结果为 `beta=0.2`。

分别报告16只建模时间测试动物、8只未见时间测试动物以及合并测试结果：

```bat
python run.py evaluate
```

运行折内相关性稳健的模块效应分析：

```bat
python run.py stats
```

导出原稿 AWA 模型真正使用的器官聚合权重：

```bat
python run.py awa-weights
```

AWA权重和SHAP贡献是两种不同的量，不应混合解释。

## 7. 完整复现 16 个模型

```bat
python run.py train --profile paper --protocol manuscript_protocol --processed-dir results/processed
```

该步骤包含 8 种结构 × 2 种训练策略 × 8 个 beta × 10 折，共 1,280 次训练，耗时最长。程序会持续写入 `results/deep/`，中断后可使用相同命令并加 `--resume` 跳过已完成的组合。

如不重训 16 种配置，可直接用仓库内已归档的 160 条入选折结果复算论文 ANOVA：

```bat
python run.py anova --fold-results reference/published_fold_metrics.csv
```

完成后复算 beta 选择、统计结果和论文指标：

```bat
python run.py select --fold-results results/deep/fold_metrics_all_betas.csv
python run.py stats --fold-results results/deep/fold_metrics.csv
python run.py verify --deep-results results/deep/model_summary.csv
```

## 8. 复现 SHAP 数值结果

先用仓库归档值快速生成数值表：

```bat
python run.py shap
```

从模型重新计算（耗时较长）：

```bat
python run.py shap --recompute --checkpoint reference/mico_pminet_fold5.pt
```

输出只包含器官和波数重要性 CSV，不生成图片。

仓库自带的SHAP缓存和重新计算命令均对应论文最终使用的 `OSB-WMHA-AWA-MORM` 模型。

## 9. 上传 GitHub

1. 在 GitHub 新建空仓库，不要自动添加 README。
2. 将本文件夹内的内容上传到仓库根目录；不要只上传外层压缩包。
3. 确认 `data/raw/all_data_acquire.csv.gz` 小于 GitHub 单文件 100 MB 限制。
4. 在公开前核对数据再分发授权，并把论文最终 DOI 补入 `CITATION.cff` 和 README。
5. 建议发布 GitHub Release `v1.3.0`，并用Zenodo归档生成永久DOI。

上传后，GitHub Actions 会自动执行快速测试和公开权重验证；绿色勾号表示其他电脑上的标准环境也通过。

如希望隔离本机 Python 环境，已安装 Docker 的电脑可执行：

```bat
docker build -t mico-pminet .
docker run --rm mico-pminet
```

## 10. 训练协议说明

默认 `manuscript_protocol` 严格按照PDF文章：每次48只训练、16只验证，最大1,000个epoch、早停200，损失仅为主预测损失和器官辅助损失（gamma=0）。`reported_results` 只用于记录旧归档权重和结果文件的来源，不是文章默认训练流程。完整差异与依据见 `ARTICLE_ALIGNMENT.md`；由于打包时没有重跑全部1,280次训练，不能把旧归档表述成新协议的现场重算结果。
