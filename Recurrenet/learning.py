# learning!

import sys
sys.path.append('..')
sys.path.insert(0, '/Users/austin/OneDrive/Repositories/')

from optimakit import SGD
from trainkit import RecurrentTrainer
from PTB import ptb # Penn Tree Bank dataset
from netkit import *
from utilkit import *


# 하이퍼파라미터 설정
batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35 # the unit for one truncated bptt
max_epoch = 4
max_grad = 0.25
eval_interval = 20

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]  # 입력
ts = corpus[1:]  # 출력（정답 레이블）

# 모델 생성
model = LSTMRecurrentNet(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(learning_rate=20.0)
trainer = RecurrentTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval)
trainer.plot(ylim=(0, 500))

model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('perplexity: ', ppl_test)

model.pkl_save_params()