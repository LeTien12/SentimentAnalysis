import json
import matplotlib.pyplot as plt


def save_json(path_file:str , file_dataset:dict):
    with open(path_file, "w") as f:
        json.dump(file_dataset, f, indent=4)
        
def load_json(file_path):

  with open(file_path, "r") as file:
      return json.load(file)
       


def draw_plot(models: dict , settings : dict):
    models['Model_Teacher'].update({"marker": "o", "color": "blue"})   
    models['Model_Strudent_ONNX'].update({"marker": "o", "color": "orange"})

    plt.figure(figsize=(9, 5))

    for label, data in models.items():
        plt.scatter(
            data["time_avg_ms"], data["accuracy"] * 100, 
            s=data["size_mb"],
            alpha=0.6, 
            marker=data["marker"], color=data["color"], label=label
        )
        plt.text(
            data["time_avg_ms"] , data["accuracy"] * 100 + 0.7, 
            f"± {data['time_std_ms']:.1f} ms", 
            fontsize=9, ha="center", color="black"
        )


    
    handles = [plt.scatter([], [], s=150, color=color['color'], alpha=0.6, label=label) 
               for label, (_ , color)in zip(models, models.items())]
    
    plt.legend(handles=handles, loc="upper right")


    plt.xlabel("Average Latency (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Performance: Latency vs Accuracy")
    plt.ylim(80, 99)  
    plt.savefig(settings.SAVE_IMG, dpi=300)
    # plt.show()


        