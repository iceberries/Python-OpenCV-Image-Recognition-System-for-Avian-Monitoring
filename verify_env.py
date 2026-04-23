"""
镜像构建完成后验证所有依赖是否完整
用法: docker run --rm bird-train:cpu python verify_env.py
"""
import sys


def check(name, import_cmd):
    try:
        exec(import_cmd)
        mod = eval(import_cmd.split("import ")[-1].split(" as ")[0])
        ver = getattr(mod, '__version__', '?')
        print(f"  [OK] {name:20s} {ver}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name:20s} {e}")
        return False


print("=" + "=" * 50)
print("  依赖完整性检查")
print("=" + "=" * 50)

results = []

results.append(check("PyTorch", "import torch"))
if 'torch' in dir():
    import torch
    print(f"       CUDA available: {torch.cuda.is_available()}")

results.append(check("torchvision", "import torchvision"))
results.append(check("NumPy", "import numpy"))
results.append(check("OpenCV", "import cv2"))
results.append(check("Pillow", "import PIL"))
results.append(check("scikit-image", "import skimage"))

# scikit-image 子模块
try:
    from skimage import exposure, transform, filters, restoration
    print(f"  [OK] {'skimage 子模块':20s} OK")
    results.append(True)
except Exception as e:
    print(f"  [FAIL] {'skimage 子模块':20s} {e}")
    results.append(False)

results.append(check("matplotlib", "import matplotlib"))
results.append(check("tqdm", "import tqdm"))
results.append(check("scipy", "import scipy"))
results.append(check("PyWavelets", "import pywt"))

# OpenCV 运行时测试
print("\n--- 运行时功能测试 ---")
try:
    import cv2, numpy as np
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    res = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    print(f"  [OK] {'OpenCV 操作':20s} resize/cvtColor/GaussianBlur")
    results.append(True)
except Exception as e:
    print(f"  [FAIL] {'OpenCV 操作':20s} {e}")
    results.append(False)

# skimage 去噪测试
try:
    from skimage import restoration
    import numpy as np
    img = np.random.rand(64, 64, 3).astype(np.float32)
    out = restoration.denoise_tv_chambolle(img, weight=0.1, channel_axis=2)
    print(f"  [OK] {'skimage TV去噪':20s} denoise_tv_chambolle (scipy OK)")
    results.append(True)
except Exception as e:
    print(f"  [FAIL] {'skimage TV去噪':20s} {e}")
    results.append(False)

try:
    from skimage import restoration
    import numpy as np
    img = np.random.rand(64, 64, 3).astype(np.float32)
    out = restoration.denoise_wavelet(img, method='BayesShrink', channel_axis=2)
    print(f"  [OK] {'小波去噪':20s} denoise_wavelet (PyWavelets OK)")
    results.append(True)
except Exception as e:
    print(f"  [FAIL] {'小波去噪':20s} {e}")
    results.append(False)

print("\n" + "=" * 52)
passed = sum(results)
total = len(results)
if passed == total:
    print(f"  ALL PASSED ({passed}/{total})")
    sys.exit(0)
else:
    print(f"  {total - passed}/{total} FAILED")
    sys.exit(1)
