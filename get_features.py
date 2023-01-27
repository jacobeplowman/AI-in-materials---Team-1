from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.structure import IStructure
import numpy as np

def get_custom_features(dataframe):
    import numpy as np
    import sys
    from atomic_number import atomic_number
    total_norm=[]
    
    surface_areas=[]
    volumes=[]
    lat_as=[]
    lat_bs=[]
    lat_cs=[]
    alphas=[]
    betas=[]
    gammas=[]
    lat_matrix_norms=[]
    crystal_systems=[]
    halls=[]
    lattice_types=[]
    is_laue=[]
    space_group_number=[]
    space_group_symbol=[]
    
    homo=[]
    lumo=[]
    band_gap_est=[]
       
    charges=[]
    densities=[]
    distance_matrix_norms=[]
    space_group_info=[]
    
    for item in dataframe.structures:
        species=[]
        for i in list(range(len(item))):
            species.append(str(item.species[i]))
        xyz = [list(a) for a in zip(species, item.cart_coords)] 
        natoms=len(item.cart_coords)
        CM = np.zeros((natoms, natoms)) 

        xyz_en=list(enumerate(xyz)) 
        for i in xyz_en:
            for j in range(natoms):
                if i[0]==j: 
                    CM[i[0]][j] = 0.5 * atomic_number[" "+i[1][0]+" "] ** 2.4  
                else: #if they are not the same atom
                    dist = np.linalg.norm(i[1][1] - xyz_en[j][1][1]) #calculating distance between atoms
                    CM[i[0]][j]= (atomic_number[" "+i[1][0]+" "]*atomic_number[" "+xyz_en[j][1][0]+" "]) / dist 

        CM_norms=[]
        for i in CM:
            CM_norms.append(np.linalg.norm(i)) 
        order=list(range(natoms)) 
        sorted_order = [list1 for i ,list1 in sorted(zip(CM_norms,order))] 
        sorted_order_desc=sorted_order[::-1] #reverses the order
        CM_sorted = np.zeros((natoms, natoms)) #creates a square array of 0's the size of natoms called CM_sorted
        xyz_en_sorted=[]
        for i in sorted_order_desc:
            xyz_en_sorted.append(xyz_en[i]) 
        counter1,counter2=0,0
        for i in xyz_en_sorted:
            for j in sorted_order_desc:
                if i[0]==j: #if the index of the atoms is the same, ie they are the same atom
                    CM_sorted[counter1][counter1] = 0.5 * atomic_number[" "+i[1][0]+" "] ** 2.4 
                    counter2=counter2+1 #adds 1 to the counter as a wey to tell which atom is currently being considered
                else:
                    dist = np.linalg.norm(i[1][1] - xyz_en_sorted[counter2][1][1])
                    CM_sorted[counter1][counter2]= (atomic_number[" "+i[1][0]+" "]*atomic_number[" "+xyz_en_sorted[counter2][1][0]+" "]) / dist #adds this to the array CM_sorted at the indices of counter1 and counter 2 which coresepond to the atoms being considered
                    counter2=counter2+1 #adds 1 to the counter as a wey to tell which atom is currently being considered
            counter2=0 #resets the counter to 0 so
            counter1=counter1+1 #adds 1 to the counter as a wey to tell which atom is currently being considered
        norm=np.linalg.norm(CM_sorted)
        total_norm.append(norm)
        

        surface_area=np.linalg.norm(np.cross(item.lattice.matrix[0], item.lattice.matrix[1]))
        surface_areas.append(surface_area)
        volumes.append(item.volume)
        lat_as.append(item.lattice.a)
        lat_bs.append(item.lattice.b)
        lat_cs.append(item.lattice.c)
        alphas.append(item.lattice.alpha)
        betas.append(item.lattice.beta)
        gammas.append(item.lattice.gamma)
        lat_matrix_norms.append(np.linalg.norm(item.lattice.matrix))
        
        sp_gr_an=SpacegroupAnalyzer(item)
        crystal_systems.append(sp_gr_an.get_crystal_system())
        halls.append(sp_gr_an.get_hall())
        lattice_types.append(sp_gr_an.get_lattice_type())
        is_laue.append(1 if sp_gr_an.is_laue()==True else 0)
        space_group_number.append(sp_gr_an.get_space_group_number())
        space_group_symbol.append(sp_gr_an.get_space_group_symbol())
        
        
        i_stru=IStructure(item.lattice,item.species,item.cart_coords)
        charges.append(i_stru.charge)
        densities.append(i_stru.density)
        distance_matrix_norms.append(np.linalg.norm(i_stru.distance_matrix))
        #space_group_info.append(i_stru.get_space_group_info())
        
        
    from matminer.featurizers.composition.orbital import AtomicOrbitals
    for item in dataframe.composition.values:
        ao=AtomicOrbitals().featurize(semiconductors.composition.values[0])
        homo.append(ao[2])
        lumo.append(ao[5])
        band_gap_est.append(ao[6])
    
    dataframe=dataframe.assign(homo=homo)
    dataframe=dataframe.assign(lumo=lumo)
    dataframe=dataframe.assign(band_gap_est=band_gap_est)
        
        
    dataframe=dataframe.assign(CM_matrix_norms=total_norm)
    
    dataframe=dataframe.assign(surface_areas=surface_areas)
    dataframe=dataframe.assign(volumes=volumes)
    dataframe=dataframe.assign(lat_as=lat_as)
    dataframe=dataframe.assign(lat_bs=lat_bs)
    dataframe=dataframe.assign(lat_cs=lat_cs)
    dataframe=dataframe.assign(alphas=alphas)
    dataframe=dataframe.assign(betas=betas)
    dataframe=dataframe.assign(gammas=gammas)
    dataframe=dataframe.assign(lat_matrix_norms=lat_matrix_norms)
    #dataframe=dataframe.assign(crystal_systems=crystal_systems)
    #dataframe=dataframe.assign(halls=halls)
    #dataframe=dataframe.assign(lattice_types=lattice_types)
    dataframe=dataframe.assign(is_laue=is_laue)
    dataframe=dataframe.assign(space_group_number=space_group_number)

    #features.insert(221,"space_group_symbol",space_group_symbol)
    
    #features.insert(222,"charges",charges)
    #features.insert(223,"densities",densities)
    #features.insert(224,"distance_matrix_norms",distance_matrix_norms)
    #features.insert(225,"space_group_info",space_group_info)
    
    
    #d=SpacegroupAnalyzer(a[0])
    #d.get_conventional_to_primitive_transformation_matrix()
    #d.get_ir_reciprocal_mesh()
    #d.get_primitive_standard_structure()
    #d.get_refined_structure()
    #d.get_space_group_number()
    #d.get_space_group_symbol()
    #d.get_symmetrized_structure()
    #d.get_symmetry_dataset()
    #d.get_symmetry_operations()
    
    #stru=IStructure(e.lattice,species,e.cart_coords)
    #stru.as_dataframe()
    #stru.frac_coords
    #stru.get_all_neighbors(1.1)
    #stru.get_miller_index_from_site_indexes(np.array(list(range(8))))
    #stru.get_miller_index_from_site_indexes([0,1,2,3,4,5,6,7])
    #stru.get_neighbor_list(2.2)
    #stru.get_neighbors_in_shell(np.array([0,0,0]),2.2,1)
    #stru.get_orderings()
    #stru.get_primitive_structure()
    #stru.get_reduced_structure()
    #stru.get_sites_in_sphere(np.array([0,0,0]),2.2)
    #stru.get_sorted_structure()
    #stru.get_space_group_info()

    dataframe.rename(columns = {'density':'density_og'}, inplace = True)
    
    from matminer.featurizers.structure import DensityFeatures
    featurizer = DensityFeatures(desired_features=['density','vpa', 'packing fraction'])
    dataframe = featurizer.featurize_dataframe(dataframe, col_id='structure', ignore_errors=True)   # there is one error
    dataframe_labels = featurizer.feature_labels()
    dataframe = dataframe.dropna(how='any', axis=0)
    
    
    return dataframe
    


def get_features_formula(formulas,space_group_symbol):
    import pandas as pd
    import numpy as np
    
    total_atom_list=[]
    for item in formulas:
        item_list=[]
        for key in (item.keys()):
            atom_list=[]
            atom_list.append(key)
            atom_list.append(item.get(key))
            item_list.append(atom_list)
        total_atom_list.append(item_list)

    unique_atoms=[]
    total_atoms=[]
    for item in total_atom_list:
        total_item=0
        unique_atoms.append(len(item))
        for atom in item:
            total_item =total_item+ atom[1]
        total_atoms.append(total_item)
        
    element_properties = pd.read_excel('element_properties.xlsx')
    total_property_list=[]
    for item in total_atom_list:
        item_list=[]
        for atom in item:
            for i in list(range(int(atom[1]))):
                item_list.append(element_properties[element_properties['Elements']==atom[0]].values)
        total_property_list.append(item_list)
        
    extended_property=[]
    for item in total_property_list:
        item_list=[]
        for atom in item:
            atom_list=[]
            for prop in atom:
                for value in prop:
                    if type(value) != str:
                        atom_list.append(value)
            item_list.append(atom_list)
        extended_property.append(item_list)
    
    label=pd.read_excel('label_new.xlsx')
    
    extended_property_real=[]
    for item in extended_property:
        item_list=[]
        for i in list(range(51)):
            prop_list=[]
            for j in list(range(len(item))):
                prop_list.append(item[j][i])
            item_list.append(sum(prop_list))
            item_list.append(min(prop_list))
            item_list.append(max(prop_list))
            item_list.append(max(prop_list)-min(prop_list))
        extended_property_real.append(item_list)


    features=pd.DataFrame(extended_property_real)
    features.insert(204,"204",unique_atoms)
    features.insert(205,"205",total_atoms)
    features=features.to_numpy()
    features=pd.DataFrame(features,columns=[label])
    
    space_group_symbols=space_group_symbol
    #features.insert(206,"space_group_symbols",space_group_symbols)
    
    return features



def get_features_structure(structures):
    import pandas as pd
    import numpy as np
    
    stru_total_atom_list=[]

    for item in structures:
        item_list=[]
        for i in list(range(len(item))):
            item_list.append(str(item.species[i]))
        stru_total_atom_list.append(item_list)
    
    stru_unique_atoms=[]
    stru_total_atoms=[]
    for item in stru_total_atom_list:
        stru_unique_atoms.append(len(set(item)))
        stru_total_atoms.append(len(item))
            
    element_properties = pd.read_excel('element_properties.xlsx')
    stru_total_property_list=[]
    for item in stru_total_atom_list:
        item_list=[]
        for atom in item:
            item_list.append(element_properties[element_properties['Elements']==atom].values)
        stru_total_property_list.append(item_list)
        
    stru_extended_property=[]
    for item in stru_total_property_list:
        item_list=[]
        for atom in item:
            atom_list=[]
            for prop in atom:
                for value in prop:
                    if type(value) != str:
                        atom_list.append(value)
            item_list.append(atom_list)
        stru_extended_property.append(item_list)
        
    label=pd.read_excel('label_new.xlsx')
    
    stru_extended_property_real=[]
    for item in stru_extended_property:
        item_list=[]
        for i in list(range(51)):
            prop_list=[]
            for j in list(range(len(item))):
                prop_list.append(item[j][i])
            item_list.append(sum(prop_list))
            item_list.append(min(prop_list))
            item_list.append(max(prop_list))
            item_list.append(max(prop_list)-min(prop_list))
        stru_extended_property_real.append(item_list)

    features=pd.DataFrame(stru_extended_property_real)
    features.insert(204,"204",stru_unique_atoms)
    features.insert(205,"205",stru_total_atoms)
    features=features.to_numpy()
    features=pd.DataFrame(features,columns=[label])
    
    return features
    
def get_features_mag(structures,col_id):
    from matminer.featurizers.composition import ElementProperty
    ep_feat_mag = ElementProperty.from_preset(preset_name="magpie")
    X_desc_mag = ep_feat_mag.featurize_dataframe(structures, col_id=col_id)
    X_desc_mag = X_desc_mag.loc[(X_desc_mag!=0).any(1), (X_desc_mag!=0).any(0)]
    X_desc_mag = X_desc_mag.dropna(how='any',axis=1)
    return X_desc_mag
    
    
def get_features_mat(structures,col_id):
    from matminer.featurizers.composition import ElementProperty
    ep_feat_mat = ElementProperty.from_preset(preset_name="matminer")
    X_desc_mat = ep_feat_mat.featurize_dataframe(structures, col_id=col_id)
    X_desc_mat = X_desc_mat.loc[(X_desc_mat!=0).any(1), (X_desc_mat!=0).any(0)]
    X_desc_mat = X_desc_mat.dropna(how='any',axis=1)
    return X_desc_mat
    
def get_features_deml(structures,col_id):
    from matminer.featurizers.composition import ElementProperty
    ep_feat_deml = ElementProperty.from_preset(preset_name="deml")
    X_desc_deml = ep_feat_deml.featurize_dataframe(structures, col_id=col_id)
    X_desc_deml = X_desc_deml.loc[(X_desc_deml!=0).any(1), (X_desc_deml!=0).any(0)]
    X_desc_deml = X_desc_deml.dropna(how='any',axis=1)
    return X_desc_deml
    

def get_features_schol(structures,col_id):
    from matminer.featurizers.composition import ElementProperty
    ep_feat_schol = ElementProperty.from_preset(preset_name="matscholar_el")
    X_desc_schol = ep_feat_schol.featurize_dataframe(structures, col_id=col_id)
    X_desc_schol = X_desc_schol.loc[(X_desc_schol!=0).any(1), (X_desc_schol!=0).any(0)]
    X_desc_schol = X_desc_schol.dropna(how='any',axis=1)
    return X_desc_schol

def get_features_megnet(structures,col_id):
    from matminer.featurizers.composition import ElementProperty
    ep_feat_megnet = ElementProperty.from_preset(preset_name="megnet_el")
    X_desc_megnet = ep_feat_megnet.featurize_dataframe(structures, col_id=col_id)
    X_desc_megnet = X_desc_megnet.loc[(X_desc_megnet!=0).any(1), (X_desc_megnet!=0).any(0)]
    X_desc_megnet = X_desc_megnet.dropna(how='any',axis=1)
    return X_desc_megnet
    
    




