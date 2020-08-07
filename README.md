# Knowledge_distillation_via_TF2.0
- **Now, I'm fixing all the issues and refining the codes. It will be easier to understand how each KD works than before.**
- **Algorithms are already implemented again, but they should be checked more with hyperparameter tuning.**
  - the algorithms which have experimental results have been confirmed.
- This Repo. will be upgraded version of my previous benchmark Repo. ([link](https://github.com/sseung0703/KD_methods_with_TF))
  
# Implemented Knowledge Distillation Methods
Defined knowledge by the neural response of the hidden layer or the output layer of the network
- Soft-logit : The first knowledge distillation method for deep neural network. Knowledge is defined by softened logits. Because it is easy to handle it, many applied methods were proposed using it such as semi-supervised learning, defencing adversarial attack and so on.
  - [Geoffrey Hinton, et al. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.](https://arxiv.org/abs/1503.02531)
- Deep Mutual Learning (DML) : train teacher and student network coincidently, to follow not only training results but teacher network's training procedure.
  - [Zhang, Ying, et al. "Deep mutual learning." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018.](http://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.html) (on worning)
- Factor Transfer (FT) : Encode a teacher network's feature map, and transfer the knowledge by mimicking it.
  - [Jangho Kim et al. "Paraphrasing Complex Network: Network Compression via Factor Transfer" Advances in Neural Information Processing Systems (NeurIPS) 2018](https://papers.nips.cc/paper/7541-paraphrasing-complex-network-network-compression-via-factor-transfer) (on worning)
Increase the quantity of knowledge by sensing several points of the teacher network
- FitNet : To increase amounts of information, knowledge is defined by multi-connected networks and compared feature maps by L2-distance.
  - [Adriana Romero, et al. Fitnets: Hints for thin deep nets. arXiv preprint arXiv:1412.6550, 2014.](https://arxiv.org/abs/1412.6550)
- Attention transfer (AT) : Knowledge is defined by attention map which is L2-norm of each feature point.
  - [Zagoruyko, Sergey et. al. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. arXiv preprint arXiv:1612.03928, 2016.](https://arxiv.org/pdf/1612.03928.pdf) [[the original project link](https://github.com/szagoruyko/attention-transfer)]
- Activation boundary (AB) : To soften teacher network's constraint, they propose the new metric function inspired by hinge loss which usually used for SVM.
  - [Byeongho Heo, et. al. Knowledge transfer via distillation of activation boundaries formed by hidden neurons. AAAI2019](https://arxiv.org/abs/1811.03233) (rivised by Author) [[the original project link](https://github.com/bhheo/AB_distillation)]
- VID : Define variational lower boundary as the knowledge, to maximize mutual information between teacher and student network. 
  - [Ahn, et. al. Variational Information Distillation for Knowledge Transfer](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf) (on worning)
Defined knowledge by the shared representation between two feature maps
- Flow of Procedure (FSP) : To soften teacher network's constraint, they define knowledge as relation of two feature maps.
  - [Junho Yim, et. al. A gift from knowledge distillation:
Fast optimization, network minimization, and transfer learning. CVPR 2017.](http://openaccess.thecvf.com/content_cvpr_2017/html/Yim_A_Gift_From_CVPR_2017_paper.html)
- KD using Singular value decomposition(KD-SVD) : To extract major information in feature map, they use singular value decomposition.
  - [Seung Hyun Lee, et. al. Self-supervised knowledge distillation using singular value decomposition. ECCV 2018](http://openaccess.thecvf.com/content_ECCV_2018/html/SEUNG_HYUN_LEE_Self-supervised_Knowledge_Distillation_ECCV_2018_paper.html) [[the original project link](https://github.com/sseung0703/SSKD_SVD)]
Defined knowledge by intra-data relation
- Relational Knowledge Distillation (RKD): they propose knowledge which contains not only feature information but also intra-data relation information.
  - [Wonpyo Park, et. al. Relational Knowledge Distillation. CVPR2019](https://arxiv.org/abs/1904.05068?context=cs.LG) [[the original project link](https://github.com/lenscloth/RKD)]
- Multi-head Graph Distillation (MHGD): They proposed the distillation module which built with the multi-head attention network. 
Each attention-head extracts the relation of feature map which contains knowledge about embedding procedure.
  - [Seunghyun Lee, Byung Cheol Song. Graph-based Knowledge Distillation by Multi-head Attention Network. BMVC2019](https://arxiv.org/abs/1907.02226) [[the original project link](https://github.com/sseung0703/MHGD)]
- Comprehensive overhaul (CO): 
  - [Heo, Byeongho, et al. A comprehensive overhaul of feature distillation. ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.html) [[the original project link](https://github.com/clovaai/overhaul-distillation/)]

# Experimental Results
- I use WResNet-40-4 and WResNet-16-4 as the teacher and the student network, respectively.
- All the algorithm is trained in the sample configuration, which is described in "train_w_distillation.py", and only each algorithm's hyper-parameters are tuned. I tried only several times to get acceptable performance, which means that my experimental results are perhaps not optimal.
- Although some of the algorithms used soft-logits parallelly in the paper, I used only the proposed knowledge distillation algorithm to make a fair comparison.
- Initialization-based methods give a far higher performance in the start point but a poor performance in the last point due to overfitting. Therefore, initialized students must have a regularization algorithm, such as Soft-logits.

## Training/Validation accuracy

|             |  Full Dataset |  50% Dataset  |  25% Dataset  |  10% Dataset  |
|:-----------:|:-------------:|:-------------:|:-------------:|:-------------:|
|   Methods   | Accuracy | Last Accuracy | Last Accuracy | Last Accuracy |
|   Teacher   |    78.59 |       -       |       -       |       -       |
|   Student   |    76.25 |       -       |       -       |       -       |
| Soft_logits |    76.57 |       -       |       -       |       -       |
|   FitNet    |    75.78 |       -       |       -       |       -       |
|      AT     |    78.14 |       -       |       -       |       -       |
|     FSP     |    76.08 |       -       |       -       |       -       |
|     DML     |        - |       -       |       -       |       -       |
|    KD_SVD   |        - |       -       |       -       |       -       |
|      FT     |    77.30 |       -       |       -       |       -       |
|      AB     |    76.52 |       -       |       -       |       -       |
|     RKD     |    77.69 |       -       |       -       |       -       |
|     VID     |        - |       -       |       -       |       -       |
|     MHGD    |        - |       -       |       -       |       -       |
|      CO     |    78.54 |       -       |       -       |       -       |

# Plan to do
- Check all the algorithms.
- do experiments.
