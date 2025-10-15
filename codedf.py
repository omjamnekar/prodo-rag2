# from transformers import AutoModel, AutoTokenizer
# # import torch

# # Model to export
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# # Dummy input
# dummy_input = tokenizer("print('hello world')", return_tensors="pt")

# # Export ONNX
# torch.onnx.export(
#     model,
#     (dummy_input["input_ids"], dummy_input["attention_mask"]),
#     "model.onnx",  # save in your project root
#     input_names=["input_ids", "attention_mask"],
#     output_names=["last_hidden_state"],
#     dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
#                   "attention_mask": {0: "batch", 1: "sequence"}},
#     opset_version=14
# )
# print("ONNX model exported: model.onnx")
