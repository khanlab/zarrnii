import typer



"""
#------------- scratch paste here

#write stardist binary label as multiscale fieldfrac instead:

#write to ome-zarr:
import pandas as pd
from ome_zarr.writer import write_image
from ome_zarr.scale import Scaler

in_img_zarr = zarr_abeta
level=0
in_label_zarr = 'stardisk_labels.zarr'
out_ome_zarr = 'img_with_fieldfrac.ome.zarr'

#read labels
darr_labels = da.from_zarr('stardisk_labels.zarr',component='/')

#read coordinate transform from ome-zarr
zi = zarr.open(in_img_zarr)
attrs=zi['/'].attrs.asdict()
multiscale=0 #first multiscale image
multiscale_datasets = attrs['multiscales'][multiscale]['datasets']
max_layer=len(multiscale_datasets)

print(multiscale_datasets)
print(max_layer)

#for writing metadata:
axes = attrs['multiscales'][multiscale]['axes']
axes = axes[-3:] #keep only the zyx axes, not channel if it exists

coordinate_transformations = [ multiscale_datasets[level]['coordinateTransformations'] for level in range(level,max_layer)]


#write ome-zarr metadata
store = zarr.DirectoryStore(out_ome_zarr,dimension_separator='/')


root = zarr.group(store,path='/',overwrite=True)

#then we add in labels but not as labels, just an image with float
scaler = Scaler(max_layer=max_layer-level-1,method='local_mean')


with ProgressBar():
    #zarr.copy_all(root_in,root)
    write_image(image=darr_labels.astype('float16'),
                            group=root,
                            scaler=scaler,
                            coordinate_transformations=coordinate_transformations,
                            storage_options={'dimension_separator': '/'},
                            axes=axes)


#  could make a script zarr to label zarr -- add label to ome zarr
 #- takes in img ome-zarr, label zarr, writes img+label ome zarr
  # opt params for label level 
  # could take in nifti label too
  #

  #for each label provide: image (nii or zarr), label_tsv (bids format), ref zarr
 #could also take in images not just labels-- in this case would need to name them and put in group
        

add_label
 in_zarr
 out_zarr (can do in-place if same as in_zarr)
 image (nii, or zarr)
 label_tsv 
 ds level 

add_dataset (?)


 add_label_to_ome_zarr


#is this tool really needed??
# would mean I would refactor existing workflows.. 

# what is the minimum to provide for users of the bids datasets:

#   resources: all relevant git repos, websites etc..
#
#   setting up python venv to work with data:
#       #make a project with dependencies to install - perhaps just setup.py
#
#   ome_zarr command-line tool
#
#   how to visualize the data:  napari <zarr>  - fiji should also work with bdv (install seems broken on cbs)
#
#   how to access the data programatically:  example getting chunks, reading attrs   
#
#   example notebook: simple thresholding to produce labels
#       load data with zarr
#       load dask array from zarr
#       first, use level=4
#           get slice
#           threshold and visualize
#       perform thresholding, save as new dask array
#       save back into ome_zarr (*) maybe this could use a helper function/library
#       save as nifti (this could similarly use a helper function/library)
#
#   manual labelling:
#       put together some various options ? 
#       example getting chunks and writing each to nifti, then back
#
#   other tools/packages:
#   nyxus - feature extraction









#--------------------
"""

app = typer.Typer()


@app.callback()
def callback():
    """
    Awesome Portal Gun
    """


@app.command()
def shoot():
    """
    Shoot the portal gun
    """
    typer.echo("Shooting portal gun")


@app.command()
def load():
    """
    Load the portal gun
    """
    typer.echo("Loading portal gun")
