"""测试TensorBoard是否可用"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 测试1：导入TensorBoard
print("="*80)
print("测试1: 导入TensorBoard")
print("="*80)
try:
    from torch.utils.tensorboard import SummaryWriter
    print("✓ TensorBoard导入成功")
except ImportError as e:
    print(f"✗ TensorBoard导入失败: {e}")
    sys.exit(1)

# 测试2：创建TensorBoard writer
print("\n" + "="*80)
print("测试2: 创建TensorBoard Writer")
print("="*80)
test_dir = "test_tensorboard_logs"
try:
    writer = SummaryWriter(test_dir)
    print(f"✓ TensorBoard Writer创建成功: {test_dir}")
    writer.close()
    print("✓ TensorBoard Writer关闭成功")
except Exception as e:
    print(f"✗ TensorBoard Writer创建失败: {e}")
    sys.exit(1)

# 测试3：写入测试数据
print("\n" + "="*80)
print("测试3: 写入测试数据")
print("="*80)
try:
    writer = SummaryWriter(test_dir)
    for i in range(10):
        writer.add_scalar('Loss/train', 1.0 - i*0.1, i)
        writer.add_scalar('Accuracy/val', i*0.1, i)
    writer.close()
    print("✓ 测试数据写入成功")
    print(f"查看TensorBoard: tensorboard --logdir {test_dir}")
except Exception as e:
    print(f"✗ 测试数据写入失败: {e}")
    sys.exit(1)

# 清理测试文件
import shutil
try:
    shutil.rmtree(test_dir)
    print(f"✓ 清理测试目录: {test_dir}")
except Exception as e:
    print(f"⚠ 清理测试目录失败（可忽略）: {e}")

print("\n" + "="*80)
print("所有测试通过！TensorBoard功能正常")
print("="*80)
