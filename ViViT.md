# ViViT

ViViT : A Video Vision Transformer
****Anurag Arnab..
Google Research

<aside>
💡 Video에 Vision Tranformer적용하기 위해 encoder, 특히 Multi-Head Self (dot)Attention를 **spatio(공간)-temporal(시간)으로 인수분해**

</aside>

```
**Paper : https://arxiv.org/pdf/2103.15691.pdf**
★review 1(kor) : https://sjpyo.tistory.com/87
review 2(kor) : https://greeksharifa.github.io/computer%20vision/2021/12/10/Transformer-based-Video-Models/
review 3(kor) : https://deep-learning-study.tistory.com/838
github : https://github.com/google-research/vision_transformer

===
colab : https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/vivit.ipynb?authuser=3 

Hugging face one video test https://huggingface.co/spaces/keras-io/video-transformers 
ViViT-B ImgNet pre-trained model https://github.com/mx-mark/VideoTransformer-pytorch

```

## 0. Abstract

---

## 1. I**ntroduction**

---

## 2. **Related Work**

---

## 3. Video Vision Transformers

### **3.1. Overview of Vison Transformers (ViT)**

### **3.2. Embedding video clips**

- Uniform frame sampling
- Tubelet embedding

### **3.3. Transformer Models for Video**

- Model 1 : Spatio-temporal attention
- Model 2 : Factorised encoder
- Model 3 : Factorised self-attention
- Model 4 : Factorised dot-product attention

### **3.4. Initialisation by leveraging pre-trained models**

- Positional embeddings
- Embedding weights, E
- Transformer weights for Model 3

---

## 4. **Expirical evaluation**

### **4.1. Experimental setup**

- Network architecture and training
- Datasets
- Inference

### **4.2. Ablation study**

- Input encoding
- Model variants
- Model regularisation
- Varying the backbone
- Varying the number of tokens
- Varying the number of input frames

### **4.3. Comparision to state-of-the-art**

- Kinetics
- Moments in Time
- Epic Kitchens 100
- Something-Something v2(SSv2)

---

## 5. **Conclusion and Future work**

---

## 6. **Appendix**

### 6.A Additional experimental details

- A.1 Further details about regularisers
    - Stochastic depth
    - Random argument
    - Label smoothing
    - Mixup
- A.2 Training hyperparameters