
from models.GAT import GAT
from models.MLP import MLP
from models.GAT_custom import GAT as GAT_custom
from models.MLP_attention import MLP_attention as AttnMLP
from models.DeepGraphConv import DeepGraphConv_Surv as DeepGraphConv
from models.PatchGCN import PatchGCN

def model_selection(Argument):

    if Argument.model == "GAT":
        model = GAT(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "GAT_custom":
        model = GAT_custom(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "MLP":
        model = MLP(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "AttMLP":
        model = AttnMLP(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "PatchGCN":
        model = PatchGCN(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    elif Argument.model == "DeepGraphConv":
        model = DeepGraphConv(Argument.dropout_rate, Argument.dropedge_rate, Argument)
    else:
        print("Enter the valid model type")
        model = None

    return model