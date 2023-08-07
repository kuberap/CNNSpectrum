
CHANNEL_LENGTH = 1024

# dictionary with paths
LABEL_DICT = {'/Data/zdroj_co': 0,
              '/Data/zdroj_cs': 1,
              '/Data/zdroj_eu_152': 2,
              '/Data/zdroj_th_232': 3,
              '/Data/zdroj_u_238': 4,
              '/Data/zdroj_am_241': 5,
              '/Data/zdroj_co_cs': 6
              }
# maximal training epochs in hyperparameter tunning
MAX_EPOCHS = 200
BATCH_SIZE =  256 #256->test acc 97.3 # 128 =>test acc 96.7, 512->test acc 88
P_ROTATION = 0.0
LR =  0.00041252074938882393/2 # test
MOMENTUM = 0.9 #0.95
WEIGHT_DECAY = 1e-6


# Pri teto koniguraci a nejlepsim modeluz pro p=0.3 napocitam pri SGD cca 95% acc na testovacich
# Adam moc dobre nefungoval, pri same konfiguraci bylo obtizne se dostatnad 80%
# MAX_EPOCHS = 100
# BATCH_SIZE = 32
# P_ROTATION = 0.3
# LR = 0.00041252074938882393/2 # test
# MOMENTUM = 0.9 #0.95

