import os
import glob
import lxml.etree as ET
import pandas as pd

def xml_to_csv(path,data_path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (data_path+root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
      trainPath = '/home/stone/Documents/MeterImages/labels/digital/data/train_labels'
      testPath = '/home/stone/Documents/MeterImages/labels/digital/data/test_labels'
      data_paths = ['/home/stone/Documents/MeterImages/labels/digital/data/train_images/','/home/stone/Documents/MeterImages/labels/digital/data/test_images/'] 	
      for index,i in enumerate([trainPath,testPath]):
          image_path = i
          folder = os.path.basename(os.path.normpath(image_path))
          xml_df = xml_to_csv(image_path,data_paths[index])
          xml_df.to_csv('/home/stone/Documents/MeterImages/labels/digital/data/'+folder+'.csv',index=None)
          print('Succesfully converted xml to csv.')

if __name__ == "__main__":
    main()
	
