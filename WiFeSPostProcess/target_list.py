import pickle

ANU_tar=dict()

ANU_tar['object'] = ['NGC5253',
                     'NGC1522', 
                     'jwst', ]

ANU_tar['pointing'] =  ['13:39:55.95 -31:38:24.36', 
                        '04:06:07.90 -52:40:06.3' , 
                        '03:32:48.64 -27:46:26.5' , ]

ANU_tar['root_path'] = ['/media/yong-group/data1/ZZY/Data_Reduction/WiFeS_postprocess_develop/NGC5253/', 
						'/media/yong-group/data1/ZZY/Data_Reduction/WiFeS_postprocess_develop/NGC1522/',
						'/media/yong-group/data1/ZZY/Data_Reduction/WiFeS_postprocess_develop/jwst_vis/', ]

ANU_tar['proc_steps'] = [[{'step':'file_setup'   , 'run':True, },
                          {'step':'substract_sky', 'run':True, }, 
                          {'step':'combine_epoch', 'run':True, },
                          {'step':'combine_grism', 'run':True, },
                          {'step':'update_wcs'   , 'run':True, },
                          {'step':'deRedden'     , 'run':True, },
                          {'step':'gene_bbimg'   , 'run':True, },],

                         [{'step':'file_setup'   , 'run':True, },
                          {'step':'substract_sky', 'run':True, }, 
                          {'step':'combine_epoch', 'run':True, },
                          {'step':'combine_grism', 'run':True, },
                          {'step':'update_wcs'   , 'run':True, },
                          {'step':'deRedden'     , 'run':True, },
                          {'step':'gene_bbimg'   , 'run':True, },],

                         [{'step':'file_setup'   , 'run':True, },
                          {'step':'substract_sky', 'run':False, }, 
                          {'step':'combine_epoch', 'run':True, },
                          {'step':'combine_grism', 'run':True, },
                          {'step':'update_wcs'   , 'run':True, },
                          {'step':'deRedden'     , 'run':True, },
                          {'step':'gene_bbimg'   , 'run':True, },], ]



pickle.dump(ANU_tar, open('target_list.pkl', "wb"))
