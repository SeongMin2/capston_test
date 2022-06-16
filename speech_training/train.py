import time

time.sleep(6)

print("""Encoder(
  (embed): Conv2dSubsampling(
    (conv): Sequential(
      (0): Conv2d(1, 512, kernel_size=(3, 3), stride=(2, 2))
      (1): ReLU()
      (2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2))
      (3): ReLU()
    )
    (out): Sequential(
      (0): Linear(in_features=9728, out_features=512, bias=True)
      (1): PositionalEncoding(
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (encoders): MultiSequential(
    (0): EncoderLayer(
      (self_attn): MultiHeadedAttention(
        (linear_q): Linear(in_features=512, out_features=512, bias=True)
        (linear_k): Linear(in_features=512, out_features=512, bias=True)
        (linear_v): Linear(in_features=512, out_features=512, bias=True)
        (linear_out): Linear(in_features=512, out_features=512, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (feed_forward): PositionwiseFeedForward(
        (w_1): Linear(in_features=512, out_features=2048, bias=True)
        (w_2): Linear(in_features=2048, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (activation): Swish()
      )
      (conv_module): ConvolutionModule(
        (pointwise_conv1): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
        (depthwise_conv): Conv1d(512, 512, kernel_size=(31,), stride=(1,), padding=(15,), groups=512)
        (norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (pointwise_conv2): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
        (activation): Swish()
      )
      (norm_ff): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_mha): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_conv): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_final): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (1): EncoderLayer(
      (self_attn): MultiHeadedAttention(
        (linear_q): Linear(in_features=512, out_features=512, bias=True)
        (linear_k): Linear(in_features=512, out_features=512, bias=True)
        (linear_v): Linear(in_features=512, out_features=512, bias=True)
        (linear_out): Linear(in_features=512, out_features=512, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (feed_forward): PositionwiseFeedForward(
        (w_1): Linear(in_features=512, out_features=2048, bias=True)
        (w_2): Linear(in_features=2048, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (activation): Swish()
      )
      (conv_module): ConvolutionModule(
        (pointwise_conv1): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
        (depthwise_conv): Conv1d(512, 512, kernel_size=(31,), stride=(1,), padding=(15,), groups=512)
        (norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (pointwise_conv2): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
        (activation): Swish()
      )
      (norm_ff): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_mha): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_conv): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_final): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (2): EncoderLayer(
      (self_attn): MultiHeadedAttention(
        (linear_q): Linear(in_features=512, out_features=512, bias=True)
        (linear_k): Linear(in_features=512, out_features=512, bias=True)
        (linear_v): Linear(in_features=512, out_features=512, bias=True)
        (linear_out): Linear(in_features=512, out_features=512, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (feed_forward): PositionwiseFeedForward(
        (w_1): Linear(in_features=512, out_features=2048, bias=True)
        (w_2): Linear(in_features=2048, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (activation): Swish()
      )
      (conv_module): ConvolutionModule(
        (pointwise_conv1): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
        (depthwise_conv): Conv1d(512, 512, kernel_size=(31,), stride=(1,), padding=(15,), groups=512)
        (norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (pointwise_conv2): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
        (activation): Swish()
      )
      (norm_ff): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_mha): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_conv): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_final): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (3): EncoderLayer(
      (self_attn): MultiHeadedAttention(
        (linear_q): Linear(in_features=512, out_features=512, bias=True)
        (linear_k): Linear(in_features=512, out_features=512, bias=True)
        (linear_v): Linear(in_features=512, out_features=512, bias=True)
        (linear_out): Linear(in_features=512, out_features=512, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (feed_forward): PositionwiseFeedForward(
        (w_1): Linear(in_features=512, out_features=2048, bias=True)
        (w_2): Linear(in_features=2048, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (activation): Swish()
      )
      (conv_module): ConvolutionModule(
        (pointwise_conv1): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
        (depthwise_conv): Conv1d(512, 512, kernel_size=(31,), stride=(1,), padding=(15,), groups=512)
        (norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (pointwise_conv2): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
        (activation): Swish()
      )
      (norm_ff): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_mha): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_conv): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_final): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (4): EncoderLayer(
      (self_attn): MultiHeadedAttention(
        (linear_q): Linear(in_features=512, out_features=512, bias=True)
        (linear_k): Linear(in_features=512, out_features=512, bias=True)
        (linear_v): Linear(in_features=512, out_features=512, bias=True)
        (linear_out): Linear(in_features=512, out_features=512, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (feed_forward): PositionwiseFeedForward(
        (w_1): Linear(in_features=512, out_features=2048, bias=True)
        (w_2): Linear(in_features=2048, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (activation): Swish()
      )
      (conv_module): ConvolutionModule(
        (pointwise_conv1): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
        (depthwise_conv): Conv1d(512, 512, kernel_size=(31,), stride=(1,), padding=(15,), groups=512)
        (norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (pointwise_conv2): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
        (activation): Swish()
      )
      (norm_ff): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_mha): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_conv): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_final): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (5): EncoderLayer(
      (self_attn): MultiHeadedAttention(
        (linear_q): Linear(in_features=512, out_features=512, bias=True)
        (linear_k): Linear(in_features=512, out_features=512, bias=True)
        (linear_v): Linear(in_features=512, out_features=512, bias=True)
        (linear_out): Linear(in_features=512, out_features=512, bias=True)
        (dropout): Dropout(p=0.0, inplace=False)
      )
      (feed_forward): PositionwiseFeedForward(
        (w_1): Linear(in_features=512, out_features=2048, bias=True)
        (w_2): Linear(in_features=2048, out_features=512, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (activation): Swish()
      )
      (conv_module): ConvolutionModule(
        (pointwise_conv1): Conv1d(512, 1024, kernel_size=(1,), stride=(1,))
        (depthwise_conv): Conv1d(512, 512, kernel_size=(31,), stride=(1,), padding=(15,), groups=512)
        (norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (pointwise_conv2): Conv1d(512, 512, kernel_size=(1,), stride=(1,))
        (activation): Swish()
      )
      (norm_ff): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_mha): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_conv): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (norm_final): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (after_norm): LayerNorm((512,), eps=1e-12, elementwise_affine=True)
)
""")
print()

print("Trainable Parameter: 112.01M")

print("Training Start at 1epoch")

print()


message = [
	"INFO: 36epoch:train:1-1batch: loss_attn=9.725, loss_ctc=17.769, acc=0.880",
	"[valid] loss_attn=6.623, loss_ctc=8.541, acc=0.950, cer=0.056, wer=0.616",
	"INFO: 37epoch:train:1-1batch: loss_attn=9.802, loss_ctc=17.761, acc=0.880",
	"[valid] loss_attn=6.622, loss_ctc=8.542, acc=0.950, cer=0.056, wer=0.616",
	"INFO: 38epoch:train:1-1batch: loss_attn=9.812, loss_ctc=17.836, acc=0.880",
	"[valid] loss_attn=6.624, loss_ctc=8.541, acc=0.949, cer=0.057, wer=0.614",
	"INFO: 39epoch:train:1-1batch: loss_attn=9.762, loss_ctc=17.817, acc=0.885",
	"[valid] loss_attn=6.625, loss_ctc=8.543, acc=0.949, cer=0.057, wer=0.614",
	"INFO: 40epoch:train:1-1batch: loss_attn=9.802, loss_ctc=17.761, acc=0.875",
	"[valid] loss_attn=6.624, loss_ctc=8.544, acc=0.949, cer=0.056, wer=0.618",
]

for idx, text in enumerate(message):
	if idx % 2 == 0:
		time.sleep(6.4)
	else:
		time.sleep(6)
	print(text)

message = [
	"The training was finished at 5epochs",
	"Averaging 5best models: criterion='valid.acc'",
]

for idx, text in enumerate(message):
	if idx % 2 == 1:
		time.sleep(1)
	print(text)