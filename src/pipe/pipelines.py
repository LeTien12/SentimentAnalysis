from src.engineer.datasets import LoadDataset
from src.settings import settings
from src.engineer import PretrainModelStudent, OnnxPipeline, ModelONNX,PretrainModelTeacher
from src.evalute import PerformanceBenchmark , OnnxPerformanceBenchmark
from src.functions_active import save_json




def pipeline() -> None:

    #Loading dataset
    dataset = LoadDataset(settings)

    # Training teacher model
    model_teacher = PretrainModelTeacher(settings=settings,datasets=dataset)

    # Training student model
    PretrainModelStudent(settings=settings,datasets=dataset)

    #Converting model student to model onnx
    ModelONNX.run(settings)

    #Creating pipeline of model onnx
    model_onnx = OnnxPipeline(settings)

    #Get path of model onnx
    model_path = model_onnx.get_model_path()

    #Evalute model teacher

    evalute_model_teacher = PerformanceBenchmark(model_teacher , dataset['test'] , optim_type="Model_Teacher")

    #Evalute model onnx

    evalute_model_onnx = OnnxPerformanceBenchmark(model_teacher , dataset['test'] , optim_type = "Model_Strudent_ONNX" , model_path=model_path)

    #Save information model

    save_json(settings.PATH_FILE_ACCURACY_TEACHER , evalute_model_teacher)
    save_json(settings.PATH_FILE_ACCURACY_STUDENT , evalute_model_onnx)
    
    






