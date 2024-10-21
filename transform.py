import pandas

foo = pandas.read_hdf("app/models/vgg16_rice_model.h5")
foo.to_json("file2.json", key="predict", mode="r")
