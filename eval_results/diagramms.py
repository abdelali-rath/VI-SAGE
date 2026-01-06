import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Globale Schriftgrößen
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14

def draw_vertical_architecture_refined():
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("Multi-Task Model Architecture", fontsize=20, fontweight='bold', pad=10)

    def draw_box(x, y, w, h, text, color, ec='black'):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=2, edgecolor=ec, facecolor=color)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=14, fontweight='bold')

    cx = 5.0 

    # 1. Input Image (Breiter & Flacher)
    in_w, in_h = 3.2, 0.7 
    draw_box(cx - in_w/2, 8.5, in_w, in_h, "Input Image", '#D1C4E9')

    # 2. Backbone (Breiter & Flacher)
    bb_w, bb_h = 4.0, 1.2
    draw_box(cx - bb_w/2, 6.0, bb_w, bb_h, "ResNet18\nBackbone", '#BBDEFB')
    
    # Pfeil Input - Backbone
    ax.annotate("", xy=(cx, 7.2), xytext=(cx, 8.5), arrowprops=dict(arrowstyle="->", lw=2, color='black'))

    # 3. Heads
    h_w, h_h = 2.8, 0.9
    h_y = 3.6
    
    draw_box(cx - 3.5 - h_w/2, h_y, h_w, h_h, "Age Head", '#C8E6C9')
    draw_box(cx - h_w/2,       h_y, h_w, h_h, "Gender Head", '#FFF9C4')
    draw_box(cx + 3.5 - h_w/2, h_y, h_w, h_h, "Ethnicity Head", '#FFCCBC')

    # 4. Wiring
    split_y = 5.2
    top_head_y = h_y + h_h 
    
    #Linien
    ax.plot([cx, cx], [6.0, split_y], color='black', lw=2)
    ax.plot([1.5, 8.5], [split_y, split_y], color='black', lw=2)
    
    # Pfeile zu den Age, Male, Groups
    for x in [1.5, 5.0, 8.5]:
        ax.annotate("", xy=(x, top_head_y), xytext=(x, split_y), arrowprops=dict(arrowstyle="->", lw=2, color='black'))

    # 5. Output Labels
    text_y = h_y - 0.5
    ax.text(1.5, text_y, "Age (Years)", ha='center', fontsize=14, fontstyle='italic')
    ax.text(5.0, text_y, "Male / Female", ha='center', fontsize=14, fontstyle='italic')
    ax.text(8.5, text_y, "5 Groups", ha='center', fontsize=14, fontstyle='italic')

    plt.savefig('architecture_vertical_refined.png', dpi=300, bbox_inches='tight')

def draw_training_plot_colored():
    epochs = np.arange(1, 13)
    train_loss = [0.7518, 0.4910, 0.3541, 0.2522, 0.1673, 0.1237, 0.0959, 0.0874, 0.0702, 0.0604, 0.0600, 0.0582]
    val_acc = [79.05, 81.50, 81.13, 79.64, 80.63, 78.93, 80.80, 81.21, 80.55, 81.00, 81.42, 80.17]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Loss
    color_loss = '#066332'
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Training Loss', color=color_loss, fontsize=16)
    l1 = ax1.plot(epochs, train_loss, color=color_loss, linewidth=3, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss, labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Accuracy
    ax2 = ax1.twinx()  
    color_acc = '#c20018'
    ax2.set_ylabel('Validation Accuracy (%)', color=color_acc, fontsize=16)
    l2 = ax2.plot(epochs, val_acc, color=color_acc, linewidth=3, marker='s', linestyle='--', label='Val Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc, labelsize=16)
    
    best_acc = val_acc[1]
    ax2.annotate(f'Best Model\n({best_acc}%)', xy=(2, best_acc), xytext=(3.5, best_acc-2),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=14, fontweight='bold')

    ax1.legend(l1 + l2, [l.get_label() for l in l1 + l2], loc='center right', fontsize=14)
    plt.title("Training Progress: Loss vs. Accuracy", fontsize=20, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('training_curve_colored.png', dpi=300)

draw_vertical_architecture_refined()
draw_training_plot_colored()