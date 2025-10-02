import matplotlib.font_manager

# 获取所有可用字体
font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# 提取字体家族名称
font_families = set()
for font in font_list:
    try:
        font_props = matplotlib.font_manager.FontProperties(fname=font)
        family_name = font_props.get_family()[0]
        font_families.add(family_name)
    except:
        continue

# 排序并打印字体家族
print("可用的字体家族 (font.family)：")
for family in sorted(font_families):
    print(f"- {family}")

# 查看中文字体（通常 macOS 系统预装的中文字体）
print("\n常见中文字体：")
chinese_fonts = [f for f in font_families if any(c in f for c in ['Heiti', 'PingFang', 'Song', 'ST', 'Sim'])]
for font in sorted(chinese_fonts):
    print(f"- {font}")


import matplotlib.pyplot as plt
import matplotlib.font_manager

# 查看所有可用字体（确认新装字体是否被识别）
font_list = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
# 搜索目标字体（如“黑体”）
print([f for f in font_list if "hei" in f.lower()])