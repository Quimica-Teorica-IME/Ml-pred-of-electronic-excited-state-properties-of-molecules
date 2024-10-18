### Avaliations functions ###
### --- importing dependences ---- ###
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb

import matplotlib.pyplot as plt
import csv
import time
import os
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.Lipinski import *
from rdkit.Chem.EState import *
from rdkit.Chem.GraphDescriptors import *
from rdkit.Chem.Graphs import *
from math import sqrt, ceil
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import numpy as np
import shap
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

### Data separation
from sklearn.preprocessing import StandardScaler




















### important functions
def grafic(y_test, y_pred, color):
    plt.plot(y_test, y_test, linestyle='--', color='black', label='y = x')

    # Plotar os pontos de teste e as previsões
    plt.scatter(y_test, y_pred, alpha=0.5, color=color, label='Pontos de Teste vs. Previsões')

    # Configurar o gráfico
    plt.title('')
    plt.xlabel('Valores reais')
    plt.ylabel('Valores preditos pelo modelo')
    plt.legend('')
    plt.grid(True)

    # Exibir o gráfico

    plt.show()
    
def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def calcular_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calcular_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def avaliar_modelo(y_true, y_pred):
    resultados = {}

    resultados['MAE'] = mean_absolute_error(y_true, y_pred)
    resultados['MSE'] = mean_squared_error(y_true, y_pred)
    resultados['RMSE'] = calcular_rmse(y_true, y_pred)
    resultados['R2'] = r2_score(y_true, y_pred)
    #resultados['MAPE'] = calcular_mape(y_true, y_pred)

    return resultados

def properties_array(sSmiles):
    try:
        m = Chem.MolFromSmiles(sSmiles)
        p1 = AllChem.GetMorganFingerprintAsBitVect(m, 2, 512)
        p2 = Chem.rdMolDescriptors.GetMACCSKeysFingerprint(m)

        p3 = [1000*FpDensityMorgan1(m), 1000*FpDensityMorgan2(m), 1000*FpDensityMorgan3(m), ExactMolWt(m), HeavyAtomMolWt(m), 1000*MaxAbsPartialCharge(m), 1000*MaxPartialCharge(m), 1000*MinAbsPartialCharge(m), 1000*MinPartialCharge(m), NumRadicalElectrons(m), NumValenceElectrons(m),1000*CalcFractionCSP3(m),10*CalcKappa1(m),10*CalcKappa2(m),10*CalcKappa3(m),CalcLabuteASA(m),CalcNumAliphaticCarbocycles(m),CalcNumAliphaticHeterocycles(m),CalcNumAliphaticRings(m),CalcNumAmideBonds(m),CalcNumAromaticCarbocycles(m),CalcNumAromaticHeterocycles(m),CalcNumAromaticRings(m),CalcNumAtomStereoCenters(m),CalcNumBridgeheadAtoms(m),CalcNumHBA(m),CalcNumHBD(m),CalcNumHeteroatoms(m),CalcNumHeterocycles(m),CalcNumLipinskiHBA(m),CalcNumLipinskiHBD(m),CalcNumRings(m),CalcNumRotatableBonds(m),CalcNumSaturatedCarbocycles(m),CalcNumSaturatedHeterocycles(m),CalcNumSaturatedRings(m),CalcNumSpiroAtoms(m),CalcNumUnspecifiedAtomStereoCenters(m),CalcTPSA(m)]
        pa3 = np.array(p3, dtype=np.int16)
        
        p4 = [HeavyAtomCount(m), NHOHCount(m), NOCount(m),NumHAcceptors(m), NumHDonors(m), Chi0(m), Chi1(m)]
        
        p5 = [rdMolDescriptors.BCUT2D(m)]

        pa1 = np.array(list(p1), dtype=np.int16)
        pa2 = np.array(list(p2), dtype=np.int16)
        pa0 = np.concatenate([pa1, pa2])
        pa4 = np.array(p4, dtype=np.int16)
        pa5 = np.array(flatten_list(p5), dtype=np.int16)
        
        pa = np.concatenate([pa0,pa3, pa4,pa5])
        #print(len(pa))

        pa = np.array(pa)

        return pa, True
    except:
        return None, False
    











### Header of data with RDKit
header3 = ['Morgan_0', 'Morgan_1', 'Morgan_2', 'Morgan_3', 'Morgan_4', 'Morgan_5', 'Morgan_6', 'Morgan_7', 'Morgan_8', 'Morgan_9', 'Morgan_10', 'Morgan_11', 'Morgan_12', 'Morgan_13', 'Morgan_14', 'Morgan_15', 'Morgan_16', 'Morgan_17', 'Morgan_18', 'Morgan_19', 'Morgan_20', 'Morgan_21', 'Morgan_22', 'Morgan_23', 'Morgan_24', 'Morgan_25', 'Morgan_26', 'Morgan_27', 'Morgan_28', 'Morgan_29', 'Morgan_30', 'Morgan_31', 'Morgan_32', 'Morgan_33', 'Morgan_34', 'Morgan_35', 'Morgan_36', 'Morgan_37', 'Morgan_38', 'Morgan_39', 'Morgan_40', 'Morgan_41', 'Morgan_42', 'Morgan_43', 'Morgan_44', 'Morgan_45', 'Morgan_46', 'Morgan_47', 'Morgan_48', 'Morgan_49', 'Morgan_50', 'Morgan_51', 'Morgan_52', 'Morgan_53', 'Morgan_54', 'Morgan_55', 'Morgan_56', 'Morgan_57', 'Morgan_58', 'Morgan_59', 'Morgan_60', 'Morgan_61', 'Morgan_62', 'Morgan_63', 'Morgan_64', 'Morgan_65', 'Morgan_66', 'Morgan_67', 'Morgan_68', 'Morgan_69', 'Morgan_70', 'Morgan_71', 'Morgan_72', 'Morgan_73', 'Morgan_74', 'Morgan_75', 'Morgan_76', 'Morgan_77', 'Morgan_78', 'Morgan_79', 'Morgan_80', 'Morgan_81', 'Morgan_82', 'Morgan_83', 'Morgan_84', 'Morgan_85', 'Morgan_86', 'Morgan_87', 'Morgan_88', 'Morgan_89', 'Morgan_90', 'Morgan_91', 'Morgan_92', 'Morgan_93', 'Morgan_94', 'Morgan_95', 'Morgan_96', 'Morgan_97', 'Morgan_98', 'Morgan_99', 'Morgan_100', 'Morgan_101', 'Morgan_102', 'Morgan_103', 'Morgan_104', 'Morgan_105', 'Morgan_106', 'Morgan_107', 'Morgan_108', 'Morgan_109', 'Morgan_110', 'Morgan_111', 'Morgan_112', 'Morgan_113', 'Morgan_114', 'Morgan_115', 'Morgan_116', 'Morgan_117', 'Morgan_118', 'Morgan_119', 'Morgan_120', 'Morgan_121', 'Morgan_122', 'Morgan_123', 'Morgan_124', 'Morgan_125', 'Morgan_126', 'Morgan_127', 'Morgan_128', 'Morgan_129', 'Morgan_130', 'Morgan_131', 'Morgan_132', 'Morgan_133', 'Morgan_134', 'Morgan_135', 'Morgan_136', 'Morgan_137', 'Morgan_138', 'Morgan_139', 'Morgan_140', 'Morgan_141', 'Morgan_142', 'Morgan_143', 'Morgan_144', 'Morgan_145', 'Morgan_146', 'Morgan_147', 'Morgan_148', 'Morgan_149', 'Morgan_150', 'Morgan_151', 'Morgan_152', 'Morgan_153', 'Morgan_154', 'Morgan_155', 'Morgan_156', 'Morgan_157', 'Morgan_158', 'Morgan_159', 'Morgan_160', 'Morgan_161', 'Morgan_162', 'Morgan_163', 'Morgan_164', 'Morgan_165', 'Morgan_166', 'Morgan_167', 'Morgan_168', 'Morgan_169', 'Morgan_170', 'Morgan_171', 'Morgan_172', 'Morgan_173', 'Morgan_174', 'Morgan_175', 'Morgan_176', 'Morgan_177', 'Morgan_178', 'Morgan_179', 'Morgan_180', 'Morgan_181', 'Morgan_182', 'Morgan_183', 'Morgan_184', 'Morgan_185', 'Morgan_186', 'Morgan_187', 'Morgan_188', 'Morgan_189', 'Morgan_190', 'Morgan_191', 'Morgan_192', 'Morgan_193', 'Morgan_194', 'Morgan_195', 'Morgan_196', 'Morgan_197', 'Morgan_198', 'Morgan_199', 'Morgan_200', 'Morgan_201', 'Morgan_202', 'Morgan_203', 'Morgan_204', 'Morgan_205', 'Morgan_206', 'Morgan_207', 'Morgan_208', 'Morgan_209', 'Morgan_210', 'Morgan_211', 'Morgan_212', 'Morgan_213', 'Morgan_214', 'Morgan_215', 'Morgan_216', 'Morgan_217', 'Morgan_218', 'Morgan_219', 'Morgan_220', 'Morgan_221', 'Morgan_222', 'Morgan_223', 'Morgan_224', 'Morgan_225', 'Morgan_226', 'Morgan_227', 'Morgan_228', 'Morgan_229', 'Morgan_230', 'Morgan_231', 'Morgan_232', 'Morgan_233', 'Morgan_234', 'Morgan_235', 'Morgan_236', 'Morgan_237', 'Morgan_238', 'Morgan_239', 'Morgan_240', 'Morgan_241', 'Morgan_242', 'Morgan_243', 'Morgan_244', 'Morgan_245', 'Morgan_246', 'Morgan_247', 'Morgan_248', 'Morgan_249', 'Morgan_250', 'Morgan_251', 'Morgan_252', 'Morgan_253', 'Morgan_254', 'Morgan_255', 'Morgan_256', 'Morgan_257', 'Morgan_258', 'Morgan_259', 'Morgan_260', 'Morgan_261', 'Morgan_262', 'Morgan_263', 'Morgan_264', 'Morgan_265', 'Morgan_266', 'Morgan_267', 'Morgan_268', 'Morgan_269', 'Morgan_270', 'Morgan_271', 'Morgan_272', 'Morgan_273', 'Morgan_274', 'Morgan_275', 'Morgan_276', 'Morgan_277', 'Morgan_278', 'Morgan_279', 'Morgan_280', 'Morgan_281', 'Morgan_282', 'Morgan_283', 'Morgan_284', 'Morgan_285', 'Morgan_286', 'Morgan_287', 'Morgan_288', 'Morgan_289', 'Morgan_290', 'Morgan_291', 'Morgan_292', 'Morgan_293', 'Morgan_294', 'Morgan_295', 'Morgan_296', 'Morgan_297', 'Morgan_298', 'Morgan_299', 'Morgan_300', 'Morgan_301', 'Morgan_302', 'Morgan_303', 'Morgan_304', 'Morgan_305', 'Morgan_306', 'Morgan_307', 'Morgan_308', 'Morgan_309', 'Morgan_310', 'Morgan_311', 'Morgan_312', 'Morgan_313', 'Morgan_314', 'Morgan_315', 'Morgan_316', 'Morgan_317', 'Morgan_318', 'Morgan_319', 'Morgan_320', 'Morgan_321', 'Morgan_322', 'Morgan_323', 'Morgan_324', 'Morgan_325', 'Morgan_326', 'Morgan_327', 'Morgan_328', 'Morgan_329', 'Morgan_330', 'Morgan_331', 'Morgan_332', 'Morgan_333', 'Morgan_334', 'Morgan_335', 'Morgan_336', 'Morgan_337', 'Morgan_338', 'Morgan_339', 'Morgan_340', 'Morgan_341', 'Morgan_342', 'Morgan_343', 'Morgan_344', 'Morgan_345', 'Morgan_346', 'Morgan_347', 'Morgan_348', 'Morgan_349', 'Morgan_350', 'Morgan_351', 'Morgan_352', 'Morgan_353', 'Morgan_354', 'Morgan_355', 'Morgan_356', 'Morgan_357', 'Morgan_358', 'Morgan_359', 'Morgan_360', 'Morgan_361', 'Morgan_362', 'Morgan_363', 'Morgan_364', 'Morgan_365', 'Morgan_366', 'Morgan_367', 'Morgan_368', 'Morgan_369', 'Morgan_370', 'Morgan_371', 'Morgan_372', 'Morgan_373', 'Morgan_374', 'Morgan_375', 'Morgan_376', 'Morgan_377', 'Morgan_378', 'Morgan_379', 'Morgan_380', 'Morgan_381', 'Morgan_382', 'Morgan_383', 'Morgan_384', 'Morgan_385', 'Morgan_386', 'Morgan_387', 'Morgan_388', 'Morgan_389', 'Morgan_390', 'Morgan_391', 'Morgan_392', 'Morgan_393', 'Morgan_394', 'Morgan_395', 'Morgan_396', 'Morgan_397', 'Morgan_398', 'Morgan_399', 'Morgan_400', 'Morgan_401', 'Morgan_402', 'Morgan_403', 'Morgan_404', 'Morgan_405', 'Morgan_406', 'Morgan_407', 'Morgan_408', 'Morgan_409', 'Morgan_410', 'Morgan_411', 'Morgan_412', 'Morgan_413', 'Morgan_414', 'Morgan_415', 'Morgan_416', 'Morgan_417', 'Morgan_418', 'Morgan_419', 'Morgan_420', 'Morgan_421', 'Morgan_422', 'Morgan_423', 'Morgan_424', 'Morgan_425', 'Morgan_426', 'Morgan_427', 'Morgan_428', 'Morgan_429', 'Morgan_430', 'Morgan_431', 'Morgan_432', 'Morgan_433', 'Morgan_434', 'Morgan_435', 'Morgan_436', 'Morgan_437', 'Morgan_438', 'Morgan_439', 'Morgan_440', 'Morgan_441', 'Morgan_442', 'Morgan_443', 'Morgan_444', 'Morgan_445', 'Morgan_446', 'Morgan_447', 'Morgan_448', 'Morgan_449', 'Morgan_450', 'Morgan_451', 'Morgan_452', 'Morgan_453', 'Morgan_454', 'Morgan_455', 'Morgan_456', 'Morgan_457', 'Morgan_458', 'Morgan_459', 'Morgan_460', 'Morgan_461', 'Morgan_462', 'Morgan_463', 'Morgan_464', 'Morgan_465', 'Morgan_466', 'Morgan_467', 'Morgan_468', 'Morgan_469', 'Morgan_470', 'Morgan_471', 'Morgan_472', 'Morgan_473', 'Morgan_474', 'Morgan_475', 'Morgan_476', 'Morgan_477', 'Morgan_478', 'Morgan_479', 'Morgan_480', 'Morgan_481', 'Morgan_482', 'Morgan_483', 'Morgan_484', 'Morgan_485', 'Morgan_486', 'Morgan_487', 'Morgan_488', 'Morgan_489', 'Morgan_490', 'Morgan_491', 'Morgan_492', 'Morgan_493', 'Morgan_494', 'Morgan_495', 'Morgan_496', 'Morgan_497', 'Morgan_498', 'Morgan_499', 'Morgan_500', 'Morgan_501', 'Morgan_502', 'Morgan_503', 'Morgan_504', 'Morgan_505', 'Morgan_506', 'Morgan_507', 'Morgan_508', 'Morgan_509', 'Morgan_510', 'Morgan_511', 'MACCS_0', 'MACCS_1', 'MACCS_2', 'MACCS_3', 'MACCS_4', 'MACCS_5', 'MACCS_6', 'MACCS_7', 'MACCS_8', 'MACCS_9', 'MACCS_10', 'MACCS_11', 'MACCS_12', 'MACCS_13', 'MACCS_14', 'MACCS_15', 'MACCS_16', 'MACCS_17', 'MACCS_18', 'MACCS_19', 'MACCS_20', 'MACCS_21', 'MACCS_22', 'MACCS_23', 'MACCS_24', 'MACCS_25', 'MACCS_26', 'MACCS_27', 'MACCS_28', 'MACCS_29', 'MACCS_30', 'MACCS_31', 'MACCS_32', 'MACCS_33', 'MACCS_34', 'MACCS_35', 'MACCS_36', 'MACCS_37', 'MACCS_38', 'MACCS_39', 'MACCS_40', 'MACCS_41', 'MACCS_42', 'MACCS_43', 'MACCS_44', 'MACCS_45', 'MACCS_46', 'MACCS_47', 'MACCS_48', 'MACCS_49', 'MACCS_50', 'MACCS_51', 'MACCS_52', 'MACCS_53', 'MACCS_54', 'MACCS_55', 'MACCS_56', 'MACCS_57', 'MACCS_58', 'MACCS_59', 'MACCS_60', 'MACCS_61', 'MACCS_62', 'MACCS_63', 'MACCS_64', 'MACCS_65', 'MACCS_66', 'MACCS_67', 'MACCS_68', 'MACCS_69', 'MACCS_70', 'MACCS_71', 'MACCS_72', 'MACCS_73', 'MACCS_74', 'MACCS_75', 'MACCS_76', 'MACCS_77', 'MACCS_78', 'MACCS_79', 'MACCS_80', 'MACCS_81', 'MACCS_82', 'MACCS_83', 'MACCS_84', 'MACCS_85', 'MACCS_86', 'MACCS_87', 'MACCS_88', 'MACCS_89', 'MACCS_90', 'MACCS_91', 'MACCS_92', 'MACCS_93', 'MACCS_94', 'MACCS_95', 'MACCS_96', 'MACCS_97', 'MACCS_98', 'MACCS_99', 'MACCS_100', 'MACCS_101', 'MACCS_102', 'MACCS_103', 'MACCS_104', 'MACCS_105', 'MACCS_106', 'MACCS_107', 'MACCS_108', 'MACCS_109', 'MACCS_110', 'MACCS_111', 'MACCS_112', 'MACCS_113', 'MACCS_114', 'MACCS_115', 'MACCS_116', 'MACCS_117', 'MACCS_118', 'MACCS_119', 'MACCS_120', 'MACCS_121', 'MACCS_122', 'MACCS_123', 'MACCS_124', 'MACCS_125', 'MACCS_126', 'MACCS_127', 'MACCS_128', 'MACCS_129', 'MACCS_130', 'MACCS_131', 'MACCS_132', 'MACCS_133', 'MACCS_134', 'MACCS_135', 'MACCS_136', 'MACCS_137', 'MACCS_138', 'MACCS_139', 'MACCS_140', 'MACCS_141', 'MACCS_142', 'MACCS_143', 'MACCS_144', 'MACCS_145', 'MACCS_146', 'MACCS_147', 'MACCS_148', 'MACCS_149', 'MACCS_150', 'MACCS_151', 'MACCS_152', 'MACCS_153', 'MACCS_154', 'MACCS_155', 'MACCS_156', 'MACCS_157', 'MACCS_158', 'MACCS_159', 'MACCS_160', 'MACCS_161', 'MACCS_162', 'MACCS_163', 'MACCS_164', 'MACCS_165', 'MACCS_166', "FpDensityMorgan1"," FpDensityMorgan2"," FpDensityMorgan3"," ExactMolWt"," HeavyAtomMolWt"," MaxAbsPartialCharge"," MaxPartialCharge"," MinAbsPartialCharge"," MinPartialCharge","NumRadicalElectrons"," NumValenceElectrons","CalcFractionCSP3","CalcKappa1","CalcKappa2","CalcKappa3","CalcLabuteASA","CalcNumAliphaticCarbocycles","CalcNumAliphaticHeterocycles","CalcNumAliphaticRings","CalcNumAmideBonds","CalcNumAromaticCarbocycles","CalcNumAromaticHeterocycles","CalcNumAromaticRings","CalcNumAtomStereoCenters","CalcNumBridgeheadAtoms","CalcNumHBA","CalcNumHBD","CalcNumHeteroatoms","CalcNumHeterocycles","CalcNumLipinskiHBA","CalcNumLipinskiHBD","CalcNumRings","CalcNumRotatableBonds","CalcNumSaturatedCarbocycles","CalcNumSaturatedHeterocycles","CalcNumSaturatedRings","CalcNumSpiroAtoms","CalcNumUnspecifiedAtomStereoCenters","CalcTPSA","HeavyAtomCount","NHOHCount","NOCount","NumHAcceptors","NumHDonors","Chi0","Chi1", 'BCUT2D_0', 'BCUT2D_1', 'BCUT2D_2','BCUT2D_3', 'BCUT2D_4', 'BCUT2D_5', 'BCUT2D_6', 'BCUT2D_7']
print(len(header3))
print(len(properties_array("C")[0]))


























#scaler = StandardScaler()

csv_file_path = "./DATA/QM-symex-modif.csv"
print('Start of training', time.ctime())
inicio = time.ctime()
pa_list = []

EE_list = []
OS_list = []
FO_list = []
SO_list = []
HOMO = []

a = 0

with open(csv_file_path , "r") as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader, None)

    rows = list(csv_reader)
    
    for row in rows:
        try:
            sSmiles = row[0]

            pa, lCano = properties_array(sSmiles)
            if not lCano:
                continue
            pa_list.append(np.concatenate([pa]))

            EE = float(row[4])
            OS = float(row[6])
            FO = float(row[8])
            SO = float(row[9])
            HO = float(row[1])
            EE_list.append(EE)
            OS_list.append(OS)
            FO_list.append(FO)
            SO_list.append(SO)
            HOMO.append(HO)
        except:
            a +=1
        #print(len(rows))
    print(f'total de erros foi: {a}')
scaler = joblib.load('./Models/scaler_model.pkl')

normalized_data = scaler.fit_transform(pa_list)
# Salvar o modelo
#joblib.dump(scaler, 'scaler_model.pkl')

without_norm = pa_list
pa_list = np.array(pa_list)

print(len(pa_list))

EE_list = np.array(EE_list)
OS_list = np.array(OS_list)
FO_list = np.array(FO_list)
SO_list = np.array(SO_list)
HOMO = np.array(HOMO)







### Avaliations functions ###
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
import numpy as np
import shap
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Função para calcular o MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Função para calcular o RMSE
def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def funcCrossValidation(model, pa, list):
    print('Cross Validation:::\n\n')

    # Data
    X, y = pa, list

    # Número de folds (k)
    num_folds = 3

    # Crie um objeto KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    # Defina os scorers para as métricas desejadas
    scoring = {
        'mse': make_scorer(mean_squared_error, greater_is_better=False),
        'mae': make_scorer(mean_absolute_error, greater_is_better=False),
        'r2': make_scorer(r2_score),
        'rmse': make_scorer(root_mean_squared_error, greater_is_better=False)
        #'mape': make_scorer(mean_absolute_percentage_error, greater_is_better=False)
    }

    # Execute a validação cruzada
    results = cross_validate(model, X, y, cv=kf, scoring=scoring, return_train_score=False)

    # Calcule a média das métricas
    mean_mse = -results['test_mse'].mean()  # Negativo porque os scorers estão invertidos
    mean_mae = -results['test_mae'].mean()  # Negativo porque os scorers estão invertidos
    mean_r2 = results['test_r2'].mean()
    mean_rmse = -results['test_rmse'].mean()  # Negativo porque os scorers estão invertidos
    #mean_mape = -results['test_mape'].mean()  # Negativo porque os scorers estão invertidos

    # Exiba as métricas médias
    print("Média do MSE:", mean_mse)
    print("Média do MAE:", mean_mae)
    print("Média do R²:", mean_r2)
    print("Média do RMSE:", mean_rmse)
    #print("Média do MAPE:", mean_mape)  

    for fold in range(num_folds):
        mse = -results['test_mse'][fold]  # Negativo porque os scorers estão invertidos
        mae = -results['test_mae'][fold]  # Negativo porque os scorers estão invertidos
        r2 = results['test_r2'][fold]
        rmse = -results['test_rmse'][fold]  # Negativo porque os scorers estão invertidos
        #mape = -results['test_mape'][fold]  # Negativo porque os scorers estão invertidos
        print(f"Fold {fold + 1}: MSE = {mse}, MAE = {mae}, R² = {r2}, RMSE = {rmse}")




def funcImportances(model,X_test):

    print('\n\n')
    # Fits the explainer
    X_test_df = pd.DataFrame(X_test[:300], columns=header3)
    explainer = shap.Explainer(model.predict, X_test_df)
    # Calculates the SHAP values 
    # Ajusta max_evals para ser no mínimo 2 * num_features + 1
    num_features = X_test_df.shape[1]
    max_evals = max(2 * num_features + 1, X_test_df.shape[1])

    shap_values = explainer(X_test_df, max_evals=max_evals)

    shap.plots.bar(shap_values)
    shap.plots.beeswarm(shap_values)



X_temp, X_test, y_temp, y_test = train_test_split(pa_list, HOMO, test_size=0.3, random_state=200)
from tensorflow import keras

# Carregar o modelo salvo
modelo = keras.models.load_model('HOMO-RNA.keras')

# Exibir a arquitetura do modelo
modelo.summary()

# Usar o modelo para fazer previsões
# Exemplo com dados fictícios
#previsoes = modelo.predict(novos_dados)

funcImportances(modelo, X_test)

