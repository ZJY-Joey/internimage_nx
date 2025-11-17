import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os


# generate color_palette.py
# ADE20K 完整 150 类别调色板 (中英对照)


plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',  # 微软雅黑 (Windows 常见)
    'SimHei',           # 黑体 (Windows 常见)
    'Arial Unicode MS', # Mac/Office 常见
    'WenQuanYi Zen Hei', # 文泉驿正黑 (Linux 常见)
    'Kaiti',            # 楷体 (通用)
    'STHeiti'           # 华文黑体 (Mac 常见)
] 
plt.rcParams['axes.unicode_minus'] = False 

color_palette_flat = [
    120,120,120, 180,120,120, 6,230,230, 80,50,50, 4,200,3, 120,120,80, 140,140,140, 204,5,255, 230,230,230, 4,250,7, 
    224,5,255, 235,255,7, 150,5,61, 120,120,70, 8,255,51, 255,6,82, 143,255,140, 204,255,4, 255,51,7, 204,70,3, 
    0,102,200, 61,230,250, 255,6,51, 11,102,255, 255,7,71, 255,9,224, 9,7,230, 220,220,220, 255,9,92, 112,9,255, 
    8,255,214, 7,255,224, 255,184,6, 10,255,71, 255,41,10, 7,255,255, 224,255,8, 102,8,255, 255,61,6, 255,194,7, 
    255,122,8, 0,255,20, 255,8,41, 255,5,153, 6,51,255, 235,12,255, 160,150,20, 0,163,255, 140,140,140, 250,10,15, 
    20,255,0, 31,255,0, 255,31,0, 255,224,0, 153,255,0, 0,0,255, 255,71,0, 0,235,255, 0,173,255, 31,0,255, 
    11,200,200, 255,82,0, 0,255,245, 0,61,255, 0,255,112, 0,255,133, 255,0,0, 255,163,0, 255,102,0, 194,255,0, 
    0,143,255, 51,255,0, 0,82,255, 0,255,41, 0,255,173, 10,0,255, 173,255,0, 0,255,153, 255,92,0, 255,0,255, 
    255,0,245, 255,0,102, 255,173,0, 255,0,20, 255,184,184, 0,31,255, 0,255,61, 0,71,255, 255,0,204, 0,255,194, 
    0,255,82, 0,10,255, 0,112,255, 51,0,255, 0,194,255, 0,122,255, 0,255,163, 255,153,0, 0,255,10, 255,112,0, 
    143,255,0, 82,0,255, 163,255,0, 255,235,0, 8,184,170, 133,0,255, 0,255,92, 184,0,255, 255,0,31, 0,184,255, 
    0,214,255, 255,0,112, 92,255,0, 0,224,255, 112,224,255, 70,184,160, 163,0,255, 153,0,255, 71,255,0, 255,0,163, 
    255,204,0, 255,0,143, 0,255,235, 133,255,0, 255,0,235, 245,0,255, 255,0,122, 255,245,0, 10,190,212, 214,255,0, 
    0,204,255, 20,0,255, 255,255,0, 0,153,255, 0,41,255, 0,255,204, 41,0,255, 41,255,0, 173,0,255, 0,245,255, 
    71,0,255, 122,0,255, 0,255,184, 0,92,255, 184,255,0, 0,133,255, 255,214,0, 25,194,194, 102,255,0, 92,0,255
]

classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']

class_translations = {
    'wall': '墙', 'building': '建筑', 'sky': '天空', 'floor': '地板', 'tree': '树', 
    'ceiling': '天花板', 'road': '道路', 'bed ': '床', 'windowpane': '窗玻璃', 'grass': '草地', 
    'cabinet': '橱柜', 'sidewalk': '人行道', 'person': '人', 'earth': '地面/泥土', 'door': '门', 
    'table': '桌子', 'mountain': '山', 'plant': '植物', 'curtain': '窗帘', 'chair': '椅子', 
    'car': '汽车', 'water': '水', 'painting': '画', 'sofa': '沙发', 'shelf': '架子', 
    'house': '房子', 'sea': '海', 'mirror': '镜子', 'rug': '地毯', 'field': '田野/场地', 
    'armchair': '扶手椅', 'seat': '座位', 'fence': '栅栏', 'desk': '书桌', 'rock': '岩石', 
    'wardrobe': '衣柜', 'lamp': '灯', 'bathtub': '浴缸', 'railing': '栏杆', 'cushion': '垫子', 
    'base': '底座', 'box': '盒子', 'column': '柱子', 'signboard': '招牌', 'chest of drawers': '抽屉柜', 
    'counter': '柜台', 'sand': '沙地', 'sink': '水槽', 'skyscraper': '摩天大楼', 'fireplace': '壁炉', 
    'refrigerator': '冰箱', 'grandstand': '看台', 'path': '路径/小路', 'stairs': '楼梯', 'runway': '跑道', 
    'case': '箱子', 'pool table': '台球桌', 'pillow': '枕头', 'screen door': '纱门', 'stairway': '楼梯间', 
    'river': '河流', 'bridge': '桥', 'bookcase': '书架', 'blind': '百叶窗', 'coffee table': '咖啡桌', 
    'toilet': '马桶', 'flower': '花', 'book': '书', 'hill': '小山', 'bench': '长凳', 
    'countertop': '台面/柜面', 'stove': '炉灶', 'palm': '棕榈树', 'kitchen island': '厨房中岛', 'computer': '电脑', 
    'swivel chair': '转椅', 'boat': '船', 'bar': '吧台', 'arcade machine': '街机', 'hovel': '窝棚', 
    'bus': '公交车', 'towel': '毛巾', 'light': '照明灯', 'truck': '卡车', 'tower': '塔', 
    'chandelier': '吊灯', 'awning': '遮阳篷', 'streetlight': '路灯', 'booth': '亭子', 'television receiver': '电视机', 
    'airplane': '飞机', 'dirt track': '土路', 'apparel': '服装', 'pole': '杆', 'land': '土地', 
    'bannister': '楼梯扶手', 'escalator': '自动扶梯', 'ottoman': '软凳', 'bottle': '瓶子', 'buffet': '餐具柜', 
    'poster': '海报', 'stage': '舞台', 'van': '面包车', 'ship': '船只', 'fountain': '喷泉', 
    'conveyer belt': '传送带', 'canopy': '顶篷', 'washer': '洗衣机', 'plaything': '玩具', 'swimming pool': '游泳池', 
    'stool': '凳子', 'barrel': '桶', 'basket': '篮子', 'waterfall': '瀑布', 'tent': '帐篷', 
    'bag': '包', 'minibike': '小型摩托车', 'cradle': '摇篮', 'oven': '烤箱', 'ball': '球', 
    'food': '食物', 'step': '台阶', 'tank': '水箱/油箱', 'trade name': '商标名', 'microwave': '微波炉', 
    'pot': '锅', 'animal': '动物', 'bicycle': '自行车', 'lake': '湖泊', 'dishwasher': '洗碗机', 
    'screen': '屏幕/屏风', 'blanket': '毯子', 'sculpture': '雕塑', 'hood': '引擎盖/罩', 'sconce': '壁灯', 
    'vase': '花瓶', 'traffic light': '交通灯', 'tray': '托盘', 'ashcan': '垃圾桶', 'fan': '风扇', 
    'pier': '码头', 'crt screen': 'CRT屏幕', 'plate': '盘子', 'monitor': '显示器', 'bulletin board': '布告板', 
    'shower': '淋浴', 'radiator': '散热器', 'glass': '玻璃杯/镜片', 'clock': '时钟', 'flag': '旗帜'
}

# --- 数据处理 ---
color_array = np.array(color_palette_flat).reshape(-1, 3) / 255.0

all_classes_info = []
for idx in range(len(classes)):
    name_en = classes[idx].strip() 
    name_cn = class_translations.get(name_en, '未知翻译')
    color = color_array[idx]
    all_classes_info.append({
        'index': idx,
        'name_en': name_en,
        'name_cn': name_cn,
        'color': color
    })

# --- 可视化 ---
num_classes = len(all_classes_info)
cols = 4 
rows = int(np.ceil(num_classes / cols))

fig, ax = plt.subplots(figsize=(15, rows * 0.75)) 

# 标题不再强制指定字体，依赖 rcParams
ax.set_title('ADE20K 完整 150 类别调色板 (中英对照)', fontsize=18, pad=20)
ax.set_xlim(0, cols)
ax.set_ylim(-rows, 0)
ax.axis('off') 

# 遍历并绘制每个类别
for i, item in enumerate(all_classes_info):
    row = i // cols
    col = i % cols
    
    x_start = col
    y_start = -row - 0.6
    
    # 绘制颜色条 (Rectangle)
    ax.add_patch(Rectangle((x_start, y_start), 0.08, 0.4, 
                           facecolor=item['color'], 
                           edgecolor='black', 
                           linewidth=0.5))
    
    # 文本内容
    label_text = (f"ID:{item['index']} | {item['name_cn']}\n"
                  f"({item['name_en']})")
    
    # 添加类别名称和索引文本
    # 文本也不再强制指定字体，依赖 rcParams
    ax.text(x_start + 0.1, y_start + 0.2, 
            label_text, 
            verticalalignment='center', 
            fontsize=8, 
            linespacing=1.2)

plt.tight_layout(rect=[0, 0.03, 1, 0.98])

# --- 保存为 PNG 文件 ---
output_filename = 'ADE20K_Full_Palette_CN.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已成功保存为文件: {os.path.abspath(output_filename)}")

plt.show()