import matplotlib.pyplot as plt
import numpy as np

# 1. Alter
labels_age = ['0-20', '21-40', '41-60', '61-80', '']
sizes_age = [26.2, 44.3, 21.1, 6.9, 1.5]
colors_age = ['#3498DB', '#1ABC9C', '#F1C40F', '#E67E22', '#C0392B']

# 2. Geschlecht
labels_gender = ['Männer', 'Frauen']
sizes_gender = [52.3, 47.7]
colors_gender = ['#28B463', '#CB4335']

# 3. Ethnie
labels_ethnicity = ['Weiße', 'Schwarz', 'Indisch', 'Asiatisch', 'Andere']
sizes_ethnicity = [42.5, 19.1, 16.8, 14.5, 7.1]
colors_ethnicity = ['#F39C12', '#8E44AD', '#D35400', '#16A085', '#7F8C8D']

# ----------------------------------------------------------------

def create_bold_pie_charts():
    # Bildgröße
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Einstellungen für Schrift
    text_style = {
        'fontsize': 25,
        'color': 'white',
        'weight': 'bold',
        'ha': 'center',
        'va': 'center'
    }

    def plot_pie(ax, sizes, labels, colors):
        wedges, texts = ax.pie(
            sizes, 
            labels=labels, 
            startangle=90, 
            colors=colors,
            labeldistance=0.5,
            textprops=text_style
        )
        ax.axis('equal')

    # Charts erstellen
    plot_pie(axes[0], sizes_age, labels_age, colors_age)
    plot_pie(axes[1], sizes_gender, labels_gender, colors_gender)
    plot_pie(axes[2], sizes_ethnicity, labels_ethnicity, colors_ethnicity)

    plt.tight_layout()
    
    # Speichern
    plt.savefig('utk_bias_white_bold.png', dpi=300, transparent=True)
    plt.show()

if __name__ == "__main__":
    create_bold_pie_charts()