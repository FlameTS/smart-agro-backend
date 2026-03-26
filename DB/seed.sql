INSERT INTO public.crops (crop_name, info) VALUES ('Apple', 'Apple is a long-living temperate fruit crop cultivated in cool to moderate climates with sufficient chilling hours. It grows best in deep') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Blueberry', 'Blueberry is a perennial fruit crop that grows best in acidic soil with a pH between 4.5 and 5.5. It requires well-drained soil rich in organic matter') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Cherry', 'Cherry is a stone fruit crop requiring fertile') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Corn', 'Corn is a major cereal crop grown worldwide under warm climatic conditions. It requires fertile soil with good moisture retention') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Grape', 'Grapevine is a perennial crop requiring warm climate') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Orange', 'Orange is a citrus crop grown in tropical and subtropical regions requiring warm temperatures and fertile soil. Trees need regular irrigation') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Peach', 'Peach is a stone fruit crop requiring well-drained soil and good air circulation. Regular pruning') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Pepper', 'Bacterial_spot') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Potato', 'Potato is a cool-season crop requiring loose') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Raspberry', 'Raspberry is a perennial fruit crop requiring well-drained soil') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Soybean', 'Soybean is a leguminous crop grown for oil and protein content. It improves soil fertility through nitrogen fixation and requires proper nutrient and water management.') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Squash', 'Squash is a warm-season vegetable crop requiring fertile soil') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Strawberry', 'Strawberry is a shallow-rooted fruit crop requiring fertile soil') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;
INSERT INTO public.crops (crop_name, info) VALUES ('Tomato', 'Tomato is a warm-season vegetable crop requiring fertile soil') ON CONFLICT (crop_name) DO UPDATE SET info = EXCLUDED.info;


DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Apple';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Apple_scab';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'fertile', cure = 'well-drained loamy soil with a slightly acidic to neutral pH. Proper orchard management including pruning' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Apple_scab', 'fertile', 'well-drained loamy soil with a slightly acidic to neutral pH. Proper orchard management including pruning');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Apple';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Black_rot';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'proper spacing', cure = 'and regular pruning. The crop is sensitive to waterlogging and benefits from good air circulation to reduce disease pressure. Timely irrigation' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Black_rot', 'proper spacing', 'and regular pruning. The crop is sensitive to waterlogging and benefits from good air circulation to reduce disease pressure. Timely irrigation');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Apple';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Cedar_apple_rust';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'soil fertility', cure = 'and effective disease prevention strategies. Trees require seasonal pruning' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Cedar_apple_rust', 'soil fertility', 'and effective disease prevention strategies. Trees require seasonal pruning');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Apple';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'climate', cure = 'and management conditions show vigorous vegetative growth and uniform fruit development. Adequate nutrient supply' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'climate', 'and management conditions show vigorous vegetative growth and uniform fruit development. Adequate nutrient supply');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Blueberry';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'consistent moisture', cure = 'and adequate sunlight. Proper mulching' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'consistent moisture', 'and adequate sunlight. Proper mulching');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Cherry';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = '(including_sour)_healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'well-drained soil and cool climatic conditions. Trees benefit from regular pruning', cure = 'proper irrigation' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, '(including_sour)_healthy', 'well-drained soil and cool climatic conditions. Trees benefit from regular pruning', 'proper irrigation');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Cherry';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = '(including_sour)_Powdery_mildew';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Powdery mildew causes white powdery growth on leaves and shoots.', cure = 'Apply suitable fungicides and remove affected plant parts.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, '(including_sour)_Powdery_mildew', 'Powdery mildew causes white powdery growth on leaves and shoots.', 'Apply suitable fungicides and remove affected plant parts.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Corn';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = '(maize)_Cercospora_leaf_spot_Gray_leaf_spot';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'adequate nitrogen supply', cure = 'and timely irrigation. Proper crop rotation' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, '(maize)_Cercospora_leaf_spot_Gray_leaf_spot', 'adequate nitrogen supply', 'and timely irrigation. Proper crop rotation');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Corn';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = '(maize)_healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'water availability', cure = 'and timely weed control contribute to vigorous growth and high productivity.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, '(maize)_healthy', 'water availability', 'and timely weed control contribute to vigorous growth and high productivity.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Corn';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = '(maize)_Northern_Leaf_Blight';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'proper spacing', cure = 'and balanced fertilizer use. Disease incidence increases under high humidity and continuous cropping without rotation.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, '(maize)_Northern_Leaf_Blight', 'proper spacing', 'and balanced fertilizer use. Disease incidence increases under high humidity and continuous cropping without rotation.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Corn';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = '(maize)Common_rust';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'resistant hybrids', cure = 'and nutrient management help reduce disease impact.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, '(maize)Common_rust', 'resistant hybrids', 'and nutrient management help reduce disease impact.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Grape';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Black_rot';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'good sunlight', cure = 'and well-drained soil. Proper training' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Black_rot', 'good sunlight', 'and well-drained soil. Proper training');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Grape';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Esca(Black_Measles)';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'soil health', cure = 'and water management. Mature vines are especially sensitive to trunk diseases.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Esca(Black_Measles)', 'soil health', 'and water management. Mature vines are especially sensitive to trunk diseases.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Grape';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'balanced vegetative growth', cure = 'and uniform fruit clusters.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'balanced vegetative growth', 'and uniform fruit clusters.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Grape';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Leaf_blight(Isariopsis_Leaf_Spot)';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Leaf blight causes brown spots leading to leaf drop.', cure = 'Apply fungicides and remove infected leaves.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Leaf_blight(Isariopsis_Leaf_Spot)', 'Leaf blight causes brown spots leading to leaf drop.', 'Apply fungicides and remove infected leaves.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Orange';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Haunglongbing(Citrus_greening)';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'balanced nutrition', cure = 'and pest control for sustained fruit production.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Haunglongbing(Citrus_greening)', 'balanced nutrition', 'and pest control for sustained fruit production.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Peach';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Bacterial_spot';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'irrigation management', cure = 'and nutrient supply are essential for healthy fruit development.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Bacterial_spot', 'irrigation management', 'and nutrient supply are essential for healthy fruit development.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Peach';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'irrigation', cure = 'and pruning practices show vigorous growth and quality fruiting.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'irrigation', 'and pruning practices show vigorous growth and quality fruiting.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Pepper';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'bell';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Bell pepper is a warm-season vegetable crop requiring fertile soil', cure = 'proper irrigation' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'bell', 'Bell pepper is a warm-season vegetable crop requiring fertile soil', 'proper irrigation');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Pepper';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'bell';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Bell pepper plants grown under optimal temperature', cure = 'moisture' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'bell', 'Bell pepper plants grown under optimal temperature', 'moisture');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Potato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Early_blight';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'fertile soil and consistent moisture. Proper crop rotation and nutrient management are essential to reduce disease incidence.', cure = 'Early blight causes brown concentric spots on leaves.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Early_blight', 'fertile soil and consistent moisture. Proper crop rotation and nutrient management are essential to reduce disease incidence.', 'Early blight causes brown concentric spots on leaves.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Potato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'soil fertility', cure = 'and disease monitoring produce healthy foliage and tubers.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'soil fertility', 'and disease monitoring produce healthy foliage and tubers.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Potato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Late_blight';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Late blight causes rapid leaf and tuber decay.', cure = 'Use resistant varieties and apply fungicides promptly.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Late_blight', 'Late blight causes rapid leaf and tuber decay.', 'Use resistant varieties and apply fungicides promptly.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Raspberry';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'moderate climate', cure = 'and regular pruning. Proper irrigation and pest control improve fruit quality.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'moderate climate', 'and regular pruning. Proper irrigation and pest control improve fruit quality.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Soybean';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Plant shows vigorous growth and green foliage.', cure = 'No treatment required; maintain soil fertility.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'Plant shows vigorous growth and green foliage.', 'No treatment required; maintain soil fertility.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Squash';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Powdery_mildew';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'full sunlight', cure = 'and proper airflow. Dense foliage increases disease risk.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Powdery_mildew', 'full sunlight', 'and proper airflow. Dense foliage increases disease risk.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Strawberry';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'proper irrigation', cure = 'and mulching. Timely nutrient supply improves yield and fruit quality.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'proper irrigation', 'and mulching. Timely nutrient supply improves yield and fruit quality.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Strawberry';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Leaf_scorch';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Leaf scorch causes purple to brown leaf margins.', cure = 'Remove infected leaves and apply fungicides.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Leaf_scorch', 'Leaf scorch causes purple to brown leaf margins.', 'Remove infected leaves and apply fungicides.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Bacterial_spot';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'adequate sunlight', cure = 'and proper spacing. Disease-free seedlings and sanitation are critical.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Bacterial_spot', 'adequate sunlight', 'and proper spacing. Disease-free seedlings and sanitation are critical.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Early_blight';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'balanced fertilization', cure = 'and proper irrigation.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Early_blight', 'balanced fertilization', 'and proper irrigation.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'healthy';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'healthy foliage', cure = 'and uniform fruiting.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'healthy', 'healthy foliage', 'and uniform fruiting.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Late_blight';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Late blight causes rapid leaf and fruit rot.', cure = 'Apply fungicides and remove infected plants.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Late_blight', 'Late blight causes rapid leaf and fruit rot.', 'Apply fungicides and remove infected plants.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Leaf_Mold';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Leaf mold causes yellow spots and fuzzy growth.', cure = 'Improve airflow and apply fungicides.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Leaf_Mold', 'Leaf mold causes yellow spots and fuzzy growth.', 'Improve airflow and apply fungicides.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Septoria_leaf_spot';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Septoria leaf spot causes small circular spots on leaves.', cure = 'Remove infected leaves and apply fungicides.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Septoria_leaf_spot', 'Septoria leaf spot causes small circular spots on leaves.', 'Remove infected leaves and apply fungicides.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Spider_mites_Two-spotted_spider_mite';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Spider mites cause yellow stippling and webbing.', cure = 'Use miticides and maintain humidity.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Spider_mites_Two-spotted_spider_mite', 'Spider mites cause yellow stippling and webbing.', 'Use miticides and maintain humidity.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Target_Spot';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Target spot causes brown circular leaf lesions.', cure = 'Apply fungicides and remove affected foliage.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Target_Spot', 'Target spot causes brown circular leaf lesions.', 'Apply fungicides and remove affected foliage.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Tomato_mosaic_virus';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Mosaic virus causes mottled leaves and stunted growth.', cure = 'Remove infected plants and disinfect tools.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Tomato_mosaic_virus', 'Mosaic virus causes mottled leaves and stunted growth.', 'Remove infected plants and disinfect tools.');
    END IF;
END $$;

DO $$
DECLARE
    v_crop_id bigint;
    v_disease_id bigint;
BEGIN
    SELECT id INTO v_crop_id FROM public.crops WHERE crop_name = 'Tomato';
    
    SELECT id INTO v_disease_id FROM public.diseases WHERE crop_id = v_crop_id AND disease_name = 'Tomato_Yellow_Leaf_Curl_Virus';
    
    IF v_disease_id IS NOT NULL THEN
        UPDATE public.diseases SET description = 'Yellow leaf curl virus causes leaf curling and yellowing.', cure = 'Control whiteflies and remove infected plants.' WHERE id = v_disease_id;
    ELSE
        INSERT INTO public.diseases (crop_id, disease_name, description, cure) VALUES (v_crop_id, 'Tomato_Yellow_Leaf_Curl_Virus', 'Yellow leaf curl virus causes leaf curling and yellowing.', 'Control whiteflies and remove infected plants.');
    END IF;
END $$;
