from rename_files import rename_original_images, rename_sliced_images
from split import slice_database
from move import move_images


rename_original_images()
slice_database()
move_images()
rename_sliced_images()
