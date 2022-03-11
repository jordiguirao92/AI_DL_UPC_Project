from fnmatch import fnmatchcase
from glob import glob
import os

source = "dataset\original\Data"

count_gt = 0
count_noisy = 0


allfiles = glob(source + "/**/*.PNG", recursive = True)
print(len(allfiles))
for file in allfiles:
    #print(file)
    if fnmatchcase(file, "*_GT_*"):
        #file = file.replace(file, os.path.join(source) + os.path.join("/") + f"image_{count_gt}_rgb.PNG")
        os.rename(file, os.path.join(source) + os.path.join("/") + f"image_{count_gt}_rgb.PNG")
        #print(file)
        count_gt = count_gt + 1
    elif fnmatchcase(file, "*_NOISY_*"):
        os.rename(file, os.path.join(source) + os.path.join("/") + f"image_{count_noisy}_rgb_noise.PNG")
        #print(file)
        count_noisy = count_noisy + 1

print(count_gt)
print(count_noisy)

#C:\Users\34619\DeepLearning\Posgraduate\AI_DL_UPC_Project\dataset\original\Data\0069_003_IP_01000_02000_3200_N
#C:\Users\34619\DeepLearning\Posgraduate\AI_DL_UPC_Project\dataset\original\Data\0149_007_G4_00800_00800_4400_L
#C:\Users\34619\DeepLearning\Posgraduate\AI_DL_UPC_Project\dataset\original\Data\0184_008_IP_00100_00100_3200_L