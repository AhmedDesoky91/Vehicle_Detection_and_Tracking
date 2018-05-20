
# coding: utf-8

# In[23]:


import os
import shutil
src = 'vehicles/vehicles/'
dest = 'vehicles_dataset/'
src_files = os.listdir(src)
full_folder_names_list = []
print(src_files)

counter = 0

for folder_name in src_files:
    if not folder_name.startswith('.'):
        full_folder_names_list.append(os.path.join(src, folder_name))
        
print(full_folder_names_list)
for folder_name in full_folder_names_list:
    files_in_folder = os.listdir(folder_name)
    print("I am in folder ", folder_name)
    print("Number of files in that folder is ", len(files_in_folder))
    for file in files_in_folder:
        full_file_name = os.path.join(folder_name, file)
        print("Source File Name ", full_file_name)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dest)
            dest_file_name = os.path.join(dest, file)
            counter = counter + 1
            end_string = '_'+str(counter)+'.png'
            print(end_string)
            new_file_name = dest_file_name.replace('.png', end_string)
            os.rename(dest_file_name, new_file_name)
            print("Dest File Name ", dest_file_name)
        else:
            print(full_file_name, " is not a file")
print("-==-==--==DONE-==--==--=")
# print(full_folder_name)
# files = os.listdir(full_folder_name[0])
# print(files)


# In[24]:


import os
import shutil
src = 'non-vehicles/non-vehicles/'
dest = 'non-vehicles_dataset/'
src_files = os.listdir(src)
full_folder_names_list = []
print(src_files)
for folder_name in src_files:
    if not folder_name.startswith('.'):
        full_folder_names_list.append(os.path.join(src, folder_name))
        
        
for folder_name in full_folder_names_list:
    files_in_folder = os.listdir(folder_name)
    for file in files_in_folder:
        full_file_name = os.path.join(folder_name, file)
        if (os.path.isfile(full_file_name)):
            shutil.copy(full_file_name, dest)

