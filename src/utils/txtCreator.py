# Create the txt file with the names
def create_txt_files(path_file, base_name, range_images, range_partition):
    with open(path_file, 'w') as f:
        for i in range(range_images[0], range_images[1]):
            for j in range(range_partition[0], range_partition[1]):
                f.write(f'{base_name}_0{i}_0{j}_rgb.png')
                f.write('\n')
                f.write(f'{base_name}_0{i}_0{j}_rgb_noise.png')
                f.write('\n')

create_txt_files("./dataset/images/training.txt", "image", [1,2], [1,8])
create_txt_files("./dataset/images/testing.txt", "image", [1,2], [8,13])