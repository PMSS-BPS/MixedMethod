import datetime
import time
from esa_snappy import ProductIO, HashMap, GPF, ProductUtils,jpy
import os, gc
from shapely import wkt
import geopandas as gpd
import json
from esa_snappy import GPF
import sys

def do_apply_orbit_file(source):
    print('\tApply orbit file...')
    parameters = HashMap()
    parameters.put('Apply-Orbit-File', True)
    parameters.put('orbitType', 'Sentinel Precise (Auto Download)')
    parameters.put('continueOnFail', True)
    output = GPF.createProduct('Apply-Orbit-File', parameters, source)
    return output

def do_thermal_noise_removal(source):
    print('\tThermal noise removal...')
    parameters = HashMap()
    parameters.put('removeThermalNoise', True)
    output = GPF.createProduct('ThermalNoiseRemoval', parameters, source)
    return output

def do_remove_grd_border_noise(source):
    print('\tRemove GRD border noise...')
    parameters = HashMap()
    parameters.put('Remove-GRD-Border-Noise', True)
    parameters.put('trimThreshold', 0.5)
    output = GPF.createProduct('Remove-GRD-Border-Noise', parameters, source)
    return output

def do_calibration(source, polarization, pols):
    print('\tCalibration...')
    parameters = HashMap()
    parameters.put('outputBetaBand', False)
    parameters.put('outputGammaBand', False)
    if polarization == 'DH':
        parameters.put('sourceBands', 'Intensity_HH,Intensity_HV')
    elif polarization == 'DV':
        parameters.put('sourceBands', 'Intensity_VH,Intensity_VV')
    elif polarization == 'SH' or polarization == 'HH':
        parameters.put('sourceBands', 'Intensity_HH')
    elif polarization == 'SV':
        parameters.put('sourceBands', 'Intensity_VV')
    else:
        print("different polarization!")
    parameters.put('selectedPolarisations', pols)
    parameters.put('outputImageScaleInDb', False)
    output = GPF.createProduct("Calibration", parameters, source)
    return output

def do_speckle_filtering(source):
    print('\tSpeckle filtering...')
    parameters = HashMap()
    parameters.put('filter', 'Refined Lee')
    output = GPF.createProduct('Speckle-Filter', parameters, source)
    return output

def do_terrain_correction(source, downsample):
    print('\tTerrain correction...')
    parameters = HashMap()
    parameters.put('demName', 'Copernicus 30m Global DEM')
    #parameters.put('demName', 'SRTM 1Sec Grid')
    parameters.put('mapProjection', 'EPSG:3857')
    parameters.put('imgResamplingMethod', 'BILINEAR_INTERPOLATION')
    parameters.put('saveProjectedLocalIncidenceAngle', False)
    parameters.put('saveSelectedSourceBand', True)
    if downsample == 1:
        parameters.put('pixelSpacingInMeter', 20.0)
    output = GPF.createProduct('Terrain-Correction', parameters, source)
    return output

def lineartodb(source):
    print('\tLinear to DB Conversion...')
    parameters = HashMap()
    output = GPF.createProduct('LinearToFromdB', parameters, source)
    return output

def preprocessing(image_in):
    print("--------------------------------------------")
    print('Preprocessing Begin')
    gc.enable()
    gc.collect()
    sentinel_1 = ProductIO.readProduct(image_in)
    print(sentinel_1)
    loopstarttime=str(datetime.datetime.now())
    print('Start time:', loopstarttime)
    start_time = time.time()
    filename=image_in.split('/')[-1].split('_')
    image_out=image_in.replace('01_Raw_Image','02_Processed_Image_rev').replace('.zip','')
    polarization=filename[3][2:]
    if polarization == 'DV':
        pols = 'VH,VV'
    elif polarization == 'DH':
        pols = 'HH,HV'
    elif polarization == 'SH' or polarization == 'HH':
        pols = 'HH'
    elif polarization == 'SV':
        pols = 'VV'
    else:
        print("Polarization error!")
    applyorbit = do_apply_orbit_file(sentinel_1)
    thermaremoved = do_thermal_noise_removal(applyorbit)
    grdborder = do_remove_grd_border_noise(thermaremoved)
    calibrated = do_calibration(grdborder, polarization, pols)
    down_speckled=do_speckle_filtering(calibrated)
    down_corrected=do_terrain_correction(down_speckled,1)
    convert=lineartodb(down_corrected)
    print("Writing...")
    ProductIO.writeProduct(convert, image_out,'BEAM-DIMAP')
    del applyorbit
    del thermaremoved
    del grdborder
    del calibrated
    del down_speckled
    del down_corrected
    sentinel_1.dispose()
    sentinel_1.closeIO()
    print("--- %s seconds ---" % (time.time() - start_time))
    print('Finshed')
    print("--------------------------------------------")

def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def mosaicing(list_sources,bound,wkt_bounds,outflnm):
    print('==========================================================')
    print('Mosaicking Process Begin')
    loopstarttime=str(datetime.datetime.now())
    print('Start time:', loopstarttime)
    start_time = time.time()
    if len(list_sources)>0:
        products = jpy.array('org.esa.snap.core.datamodel.Product',len(list_sources))
        Variable = jpy.get_type('org.esa.snap.core.gpf.common.MosaicOp$Variable')
        variables = jpy.array('org.esa.snap.core.gpf.common.MosaicOp$Variable', 2)
    
        parameters = HashMap()
        parameters.put('combine', 'OR')
        parameters.put('crs', 'EPSG:3857')
        parameters.put('resampling', 'Nearest')
        parameters.put('pixelSizeX', 20.0)
        parameters.put('pixelSizeY', 20.0)
        parameters.put('orthorectify', False)
        parameters.put('westBound',bound[0])
        parameters.put('southBound',bound[1])
        parameters.put('northBound',bound[3])
        parameters.put('eastBound',bound[2])
    
        for i in range(len(list_sources)):
            p=ProductIO.readProduct('/data/ksa/01_Image_Acquisition/02_Processed_Image_rev/'+list_sources[i]+'.dim')
            products[i]=p
        if len(list_sources)>0:
            band_names = products[0].getBandNames()
            i=0
            for band in band_names:
                print(band)
                variables[i]=Variable(band,band)
                i=i+1 
        parameters.put('variables', variables) 
        output = GPF.createProduct('Mosaic', parameters, products)
    
        parameters = HashMap()
        parameters.put('geoRegion', wkt_bounds)
        output = GPF.createProduct('Subset', parameters, output)

        ProductIO.writeProduct(output, outflnm, 'BEAM-DIMAP')
        for i in range(len(list_sources)):
            products[i].dispose()
            products[i].closeIO()
        del output
    else:
        print('Encounter missing acquisition!!!')
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Finished")
    print('==========================================================')

def run_mosaic(dict_mosaic):
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('PREPROCESSING FOLLOWED BY MOSAICKING FOR ID:',dict_mosaic['id'])
    create_folder_if_not_exist('/data/ksa/01_Image_Acquisition/02_Processed_mosaic/'+dict_mosaic['id'])
    bounds=gpd.GeoDataFrame(geometry=[wkt.loads(dict_mosaic['geometry'])],crs=4326)#.to_crs(3857)
    wkt_bounds=bounds.geometry.to_wkt()[0]
    total_bounds=bounds.total_bounds
    period_dict=dict_mosaic['periode']
    for i in range(len(period_dict)):
        start_periode=period_dict[i]['start_periode']
        end_periode=period_dict[i]['end_periode']
        print('PERIODE ',start_periode,'-',end_periode)
        outflnm='/data/ksa/01_Image_Acquisition/02_Processed_mosaic/'+dict_mosaic['id']+'/'+start_periode.replace('-','')+'_'+end_periode.replace('-','')
        list_sources= period_dict[i]['image_asf']
        list_sources_mosaic=[]
        if not os.path.exists(outflnm+'.dim'):
            for j in list_sources:
                try:
                    # 35, 16, dan 18 di reset ya.
                    #if not os.path.exists('/data/ksa/01_Image_Acquisition/02_Processed_Image_rev/'+j+'.dim'):
                    preprocessing('/data/ksa/01_Image_Acquisition/01_Raw_Image/'+j+'.zip')
                    #else:
                    #    try:
                    #        sentinel_1 = ProductIO.readProduct('/data/ksa/01_Image_Acquisition/02_Processed_Image_rev/'+j+'.dim')
                    #        print(sentinel_1)
                    #        print('Image for: ', '/data/ksa/01_Image_Acquisition/02_Processed_Image_rev/'+j+'.dim',' already processed. Skip')
                    #        sentinel_1.dispose()
                    #        sentinel_1.closeIO()
                    #    except:
                    #        preprocessing('/data/ksa/01_Image_Acquisition/01_Raw_Image/'+j+'.zip')
                    list_sources_mosaic.append(j)
                except Exception as e:
                    print("An error occurred:", e)
            mosaicing(list_sources_mosaic,total_bounds,wkt_bounds,outflnm)
        else:
            print('Mosaic file for the current period has been processed previously.')
        #    print('Encountering Error while doing the mosaic due to incompleted process previously.')
        #break
    print('FINISHED')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')  

def main():
    kdprov=sys.argv[1]
    metadata=f'/data/ksa/01_Image_Acquisition/05_Json_Coverage/{kdprov}_coverage_ASF.json'
    with open(metadata,'r') as f:
        dt_prov=json.load(f)
    for i in range(0,len(dt_prov)):
        print('*********************************************************')
        print('Preprocessing followed by mosaic begin for PROV:',kdprov)
        print(dt_prov[i]['id'])
        if dt_prov[i]['id'] in ['54LVR', '54LVS', '54MVT', '54MUT']:
            run_mosaic(dt_prov[i])
        print('*********************************************************')
        
if __name__ == "__main__":
    main()
