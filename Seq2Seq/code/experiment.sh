python -m nmt.nmt \
    --attention=bahdanau \
    --src=en --tgt=vi \
    --learning_rate=0.001 \
    --optimizer=adam \
    --vocab_prefix=./nmt_data/vocab  \
    --train_prefix=./nmt_data/train \
    --dev_prefix=./nmt_data/tst2012  \
    --test_prefix=./nmt_data/tst2013 \
    --out_dir=./nmt_attention_model \
    --num_train_steps=24000 \
    --steps_per_stats=100 \
    --num_layers=2 \
    --num_units=512 \
    --dropout=0.2 \
    --metrics=bleu \
    --encoder_type=bi \
    --batch_size=32 \
    --decay_scheme=luong234 \

# #Experiment2 :RNN type
#
# # mkdir ./nmt_gru
# #
# # python -m nmt.nmt \
# #     --attention=bahdanau \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.001 \
# #     --optimizer=adam \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_gru \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=2 \
# #     --num_units=512 \
# #     --dropout=0.2 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=32 \
# #     --unit_type=gru
# #
# # #Experiment3 : Number of layers
# # mkdir ./nmt_layer_2
# #
# # python -m nmt.nmt \
# #     --attention=bahdanau \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.001 \
# #     --optimizer=adam \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_layer_2 \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=4 \
# #     --num_units=512 \
# #     --dropout=0.2 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=32 \
# #
# # mkdir ./nmt_layer_3
# #
# #     python -m nmt.nmt \
# #         --attention=bahdanau \
# #         --src=en --tgt=vi \
# #         --learning_rate=0.001 \
# #         --optimizer=adam \
# #         --vocab_prefix=./nmt_data/vocab  \
# #         --train_prefix=./nmt_data/train \
# #         --dev_prefix=./nmt_data/tst2012  \
# #         --test_prefix=./nmt_data/tst2013 \
# #         --out_dir=./nmt_layer_3 \
# #         --num_train_steps=4500 \
# #         --steps_per_stats=100 \
# #         --num_layers=6 \
# #         --num_units=512 \
# #         --dropout=0.2 \
# #         --metrics=bleu \
# #         --encoder_type=bi \
# #         --batch_size=32 \
# #
# # #Experiment4 : Encoder, decoder layer length set to different values
# # mkdir ./nmt_enc_dec_layer
# #
# #     python -m nmt.nmt \
# #         --attention=bahdanau \
# #         --src=en --tgt=vi \
# #         --learning_rate=0.001 \
# #         --optimizer=adam \
# #         --vocab_prefix=./nmt_data/vocab  \
# #         --train_prefix=./nmt_data/train \
# #         --dev_prefix=./nmt_data/tst2012  \
# #         --test_prefix=./nmt_data/tst2013 \
# #         --out_dir=./nmt_enc_dec_layer \
# #         --num_train_steps=4500 \
# #         --steps_per_stats=100 \
# #         --num_units=512 \
# #         --dropout=0.2 \
# #         --metrics=bleu \
# #         --encoder_type=bi \
# #         --batch_size=32 \
# #         --num_encoder_layers=4 \
# #         --num_decoder_layers=3 \
# #
# # #Experiment5 : Number of units
# # mkdir ./nmt_num_units_256
# #
# #     python -m nmt.nmt \
# #         --attention=bahdanau \
# #         --src=en --tgt=vi \
# #         --learning_rate=0.001 \
# #         --optimizer=adam \
# #         --vocab_prefix=./nmt_data/vocab  \
# #         --train_prefix=./nmt_data/train \
# #         --dev_prefix=./nmt_data/tst2012  \
# #         --test_prefix=./nmt_data/tst2013 \
# #         --out_dir=./nmt_num_units_256 \
# #         --num_train_steps=4500 \
# #         --steps_per_stats=100 \
# #         --num_layers=2 \
# #         --num_units=256 \
# #         --dropout=0.2 \
# #         --metrics=bleu \
# #         --encoder_type=bi \
# #         --batch_size=32 \
# #
# # mkdir ./nmt_num_units_1024
# #
# #     python -m nmt.nmt \
# #         --attention=bahdanau \
# #         --src=en --tgt=vi \
# #         --learning_rate=0.001 \
# #         --optimizer=adam \
# #         --vocab_prefix=./nmt_data/vocab  \
# #         --train_prefix=./nmt_data/train \
# #         --dev_prefix=./nmt_data/tst2012  \
# #         --test_prefix=./nmt_data/tst2013 \
# #         --out_dir=./nmt_num_units_1024 \
# #         --num_train_steps=4500 \
# #         --steps_per_stats=100 \
# #         --num_layers=2 \
# #         --num_units=1024 \
# #         --dropout=0.2 \
# #         --metrics=bleu \
# #         --encoder_type=bi \
# #         --batch_size=32 \
# #
# # #Experiment6 : Attention mechanism
# # mkdir ./nmt_luong
# #
# # python -m nmt.nmt \
# #     --attention=luong \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.001 \
# #     --optimizer=adam \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_luong \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=2 \
# #     --num_units=512 \
# #     --dropout=0.2 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=32 \
# #
# # #Experiment7: optimizer
# # mkdir ./nmt_sgd
# #
# # python -m nmt.nmt \
# #     --attention=bahdanau \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.1 \
# #     --optimizer=sgd \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_sgd \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=2 \
# #     --num_units=512 \
# #     --dropout=0.2 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=32 \
# #
# # #Experiment8 : learning_rate
# # mkdir ./nmt_lr_0_01
# #
# # python -m nmt.nmt \
# #     --attention=bahdanau \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.01 \
# #     --optimizer=adam \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_lr_0_01 \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=2 \
# #     --num_units=512 \
# #     --dropout=0.2 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=32 \
# #
# # mkdir ./nmt_lr_0_0001
# #
# # python -m nmt.nmt \
# #     --attention=bahdanau \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.0001 \
# #     --optimizer=adam \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_lr_0_0001 \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=2 \
# #     --num_units=512 \
# #     --dropout=0.2 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=32 \
# #
# # #Experiment9 : Batch size
# # mkdir ./nmt_batch_16
# #
# # python -m nmt.nmt \
# #     --attention=bahdanau \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.001 \
# #     --optimizer=adam \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_batch_16 \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=2 \
# #     --num_units=512 \
# #     --dropout=0.2 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=16 \
# #
# # mkdir ./nmt_batch_128
# #
# # python -m nmt.nmt \
# #     --attention=bahdanau \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.001 \
# #     --optimizer=adam \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_batch_128 \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=2 \
# #     --num_units=512 \
# #     --dropout=0.2 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=128 \
# #
# # #Experiment10 : dropout
# # mkdir ./nmt_dropout_05
# #
# # python -m nmt.nmt \
# #     --attention=bahdanau \
# #     --src=en --tgt=vi \
# #     --learning_rate=0.001 \
# #     --optimizer=adam \
# #     --vocab_prefix=./nmt_data/vocab  \
# #     --train_prefix=./nmt_data/train \
# #     --dev_prefix=./nmt_data/tst2012  \
# #     --test_prefix=./nmt_data/tst2013 \
# #     --out_dir=./nmt_dropout_05 \
# #     --num_train_steps=4500 \
# #     --steps_per_stats=100 \
# #     --num_layers=2 \
# #     --num_units=512 \
# #     --dropout=0.5 \
# #     --metrics=bleu \
# #     --encoder_type=bi \
# #     --batch_size=32 \
#
# #Experiment11 : beam_width
# mkdir ./nmt_beam_3
#
# python -m nmt.nmt \
#     --attention=bahdanau \
#     --src=en --tgt=vi \
#     --learning_rate=0.001 \
#     --optimizer=adam \
#     --vocab_prefix=./nmt_data/vocab  \
#     --train_prefix=./nmt_data/train \
#     --dev_prefix=./nmt_data/tst2012  \
#     --test_prefix=./nmt_data/tst2013 \
#     --out_dir=./nmt_beam_3 \
#     --num_train_steps=4500 \
#     --steps_per_stats=100 \
#     --num_layers=2 \
#     --num_units=512 \
#     --dropout=0.2 \
#     --metrics=bleu \
#     --encoder_type=bi \
#     --batch_size=32 \
#     --beam_width=3 \
#
# mkdir ./nmt_beam_10
#
# python -m nmt.nmt \
#     --attention=bahdanau \
#     --src=en --tgt=vi \
#     --learning_rate=0.001 \
#     --optimizer=adam \
#     --vocab_prefix=./nmt_data/vocab  \
#     --train_prefix=./nmt_data/train \
#     --dev_prefix=./nmt_data/tst2012  \
#     --test_prefix=./nmt_data/tst2013 \
#     --out_dir=./nmt_beam_10 \
#     --num_train_steps=4500 \
#     --steps_per_stats=100 \
#     --num_layers=2 \
#     --num_units=512 \
#     --dropout=0.2 \
#     --metrics=bleu \
#     --encoder_type=bi \
#     --batch_size=32 \
#     --beam_width=10 \
#
# mkdir ./nmt_beam_50
#
# python -m nmt.nmt \
#     --attention=bahdanau \
#     --src=en --tgt=vi \
#     --learning_rate=0.001 \
#     --optimizer=adam \
#     --vocab_prefix=./nmt_data/vocab  \
#     --train_prefix=./nmt_data/train \
#     --dev_prefix=./nmt_data/tst2012  \
#     --test_prefix=./nmt_data/tst2013 \
#     --out_dir=./nmt_beam_50 \
#     --num_train_steps=4500 \
#     --steps_per_stats=100 \
#     --num_layers=2 \
#     --num_units=512 \
#     --dropout=0.2 \
#     --metrics=bleu \
#     --encoder_type=bi \
#     --batch_size=32 \
#     --beam_width=50 \
#
# #$Experiment13 : decay steps
# mkdir ./nmt_decay_scheme
#
# python -m nmt.nmt \
#     --attention=bahdanau \
#     --src=en --tgt=vi \
#     --learning_rate=0.001 \
#     --optimizer=adam \
#     --vocab_prefix=./nmt_data/vocab  \
#     --train_prefix=./nmt_data/train \
#     --dev_prefix=./nmt_data/tst2012  \
#     --test_prefix=./nmt_data/tst2013 \
#     --out_dir=./nmt_decay_scheme \
#     --num_train_steps=4500 \
#     --steps_per_stats=100 \
#     --num_layers=2 \
#     --num_units=512 \
#     --dropout=0.2 \
#     --metrics=bleu \
#     --encoder_type=bi \
#     --batch_size=32 \
#     --decay_scheme=luong5 \
#
# mkdir ./nmt_warmup_scheme
#
# python -m nmt.nmt \
#     --attention=bahdanau \
#     --src=en --tgt=vi \
#     --learning_rate=0.01 \
#     --optimizer=adam \
#     --vocab_prefix=./nmt_data/vocab  \
#     --train_prefix=./nmt_data/train \
#     --dev_prefix=./nmt_data/tst2012  \
#     --test_prefix=./nmt_data/tst2013 \
#     --out_dir=./nmt_warmup_scheme \
#     --num_train_steps=4500 \
#     --steps_per_stats=100 \
#     --num_layers=2 \
#     --num_units=512 \
#     --dropout=0.2 \
#     --metrics=bleu \
#     --encoder_type=bi \
#     --batch_size=32 \
#     --warmup_steps=1500 \
#     --warmup_scheme=t2t \
# #************************************Inference******************************
# # python -m nmt.nmt \
# #   --out_dir=./AWS/1/nmt_attention_model \
# #   --inference_input_file=./nmt_data/my_infer_file.en \
# #   --inference_output_file=./nmt_data/output_infer.txt
# #
# # cat ./nmt_data/output_infer.txt # To view the inference as output
