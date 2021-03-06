# Create the txt file with the names
def create_txt_files(path_file, base_name, range_images, range_partition):
    with open(path_file, 'w') as f:
        for i in range(range_images[0], range_images[1]):
            for j in range(range_partition[0], range_partition[1]):
                f.write(f'{base_name}_{i}_{j}_rgb.png')
                f.write('\n')
                #Eliminate the rgb_noise
                #f.write(f'{base_name}_{i}_{j}_rgb_noise.png')
                #f.write('\n')

create_txt_files("./dataset/images/training.txt", "image", [1,35], [1,13]) #70% All [1,223], [1,13]
create_txt_files("./dataset/images/validation.txt", "image", [35,43], [1,13]) #15% All [223,270], [1,13]
create_txt_files("./dataset/images/testing.txt", "image", [270,318], [1,13]) #15% All [1,223], [1,13]