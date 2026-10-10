import cv2,time, torch, traceback
from core.settings import Settings
from utils import UtilsPipeline, ModelLoading


env = Settings()

class PcbInspectionPipeline(ModelLoading, UtilsPipeline):
    def __init__(self , logger):
        super().__init__(logger)

        self.logger = logger
        # Models
        self.model = self.load_classifier(env)    # Classifier Model
        self.ocr_model = self.load_OCR_model(env) # OCR MODEL

        self.components_labels = {
                    "missing": ["Not Missing", "Missing"],
                    "Shifted": ["Not Shifted", "Shifted"],
                    "IC_Direction": ["Not an IC" , "IC is upwards", "IC is downwards"],
                    "Bridge": ['Ignore', "Not short", "Short"],
                    "Capacitor_Color": ["Not_Capacitor", "Brown", "YELLOW", "WHITE"]
                    }
        
        self.pin_labels = {
            "Ignore" : 0,
            "Not short": 1,
            "Short":2
        }

    def prediction(self,pred):
        preds = []
        ouptuts = []
        for head in pred:
            ouptut = pred[head].cpu().detach().numpy().argmax()
            preds.append(ouptut)

        for out , key in zip(preds, self.components_labels.keys()):
            ouptuts.append(self.components_labels[key][out])
        return ouptuts

    def extract_text(self, test_crop):
        return (self.ocr_model.ocr(img=test_crop, rec=True, det=False) , self.ocr_model.ocr(img=self.thresh(test_crop, threshold=50), rec=True, det=False))

    def component_infernece(self, test_image, true_image):
        avg = 0
        for key, value in  self.classifer_config.items():
            stime = time.time()
            try:
                true_label = value[1]
                orientaion = value[0]
             
                if len(self.components_crops[key]) == 1:
                    crop =  self.components_crops[key][0]
                    x1, y1, x2, y2 = map(round, crop)

                    test_crop =  test_image[y1:y2, x1:x2]
                    test_crop =  self.set_orientaion(test_crop,orientaion)
                    original_image = true_image[y1:y2, x1:x2]

                    cv2.imwrite("test_img.png", cv2.cvtColor(test_crop, cv2.COLOR_RGB2BGR))
                    tested_image = cv2.resize(test_crop, (128,256), interpolation=cv2.INTER_LINEAR)
                    test_crop = self.component_preprocess(tested_image, orientaion)
                    pred=self.model(test_crop)
                    result = self.prediction(pred)

                else:
                    outputs = []
                    save_image = True
                    for crop in self.components_crops[key]:
                        x1, y1, x2, y2 = map(round, crop)

                        test_crop =  test_image[y1:y2, x1:x2]
                        test_crop =  self.set_orientaion(test_crop,orientaion)

                       
                        if test_crop.shape[0] < test_crop.shape[1] and key != "U3":
                            test_crop =  self.set_orientaion(test_crop,90)

                        cv2.imwrite(f"test_img.png", cv2.cvtColor(test_crop, cv2.COLOR_RGB2BGR))

                        test_img = cv2.resize(test_crop, (128,256), interpolation=cv2.INTER_LINEAR)

                        if save_image:
                            tested_image = test_img
                            original_image = true_image[y1:y2, x1:x2]
                            save_image = False


                        test_crop = self.component_preprocess(test_img, orientaion)
                        pred=self.model(test_crop)
                        outputs.append(self.prediction(pred))

                    
                    result = outputs.pop(0)
                    bridge = 0
                    for out in outputs:
                        if bridge < self.pin_labels[out[3]]:
                            bridge = self.pin_labels[out[3]]
                    result[3] = self.components_labels["Bridge"][bridge]



                mismatches = [(index, r_val, t_val) for index, (r_val, t_val) in enumerate(zip(result, true_label)) if r_val != t_val]


                if mismatches:
                    if (0, 'Missing', 'Not Missing') in mismatches:
                        mismatches = [(0, 'Missing', 'Not Missing')]

                    for index, r_val, t_val in mismatches:
                        self.logger.log_main(f"Detect Found: Component {key} || Recognized_Defect: {r_val}  || Original_Label: {t_val}", "error")
                        original_image = self.set_orientaion(original_image, orientaion)
                        original_image = cv2.resize(original_image, (128,256), interpolation=cv2.INTER_LINEAR)
                        error = next((key for key, values in self.components_labels.items() if r_val in values), None)
                        
                        try:
                            defected_image = self.save_defect_image(original_image, tested_image, f"{error}", key)
                            self.defect_found = True
                            for num,val in  enumerate(self.defects['defects']):
                                if val['defect_type'] == error:
                                    self.defects['defects'][num]["count"] +=1
                                    self.defects['defects'][num]["data"].append({"Component": key, "image": defected_image})
                                    break
                        except:
                            import pdb;pdb.set_trace()

                else:
                   self.logger.log_main(f"{key} Component Inspection Done!")
                
                avg+= (time.time() - stime)
                   

            except Exception as e:
                traceback_str = traceback.format_exc()
                self.logger.log_exception(f"Error in Component Inspection: {key}: {traceback_str}")
        self.logger.log_time(f"Classifier Inspection Average Time of each Component:  {round(avg/len(self.classifer_config.keys()) , 3)}secs")

    def ocr_inference(self, test_image, true_image):
        avg = 0
        for key, orientaion in self.ocr_config.items():
            stime = time.time()
            try:
                original_label = orientaion[0]
                orientaion = orientaion[1]
                crop =  self.component_ocr[key]
                x1, y1, x2, y2 = map(round, crop)
                test_crop =  test_image[y1:y2, x1:x2]
                test_crop = self.set_orientaion(test_crop,orientaion)

            
                true_res,  true_res_thresh = self.extract_text(test_crop)
                if (true_res and true_res[0] and true_res[0][0]) :
                    template_text, _ = true_res[0][0]
                    template_text = template_text.strip().upper()
            
                if (true_res_thresh and true_res_thresh[0] and true_res_thresh[0][0]) :
                    template_text_thresh, _ = true_res_thresh[0][0]
                    template_text_thresh = template_text_thresh.strip().upper()
                
                if original_label != template_text and original_label != template_text_thresh:
                    test_crop = cv2.rotate(test_crop, cv2.ROTATE_180)
                    true_res,  true_res_thresh = self.extract_text(test_crop)

                if (true_res and true_res[0] and true_res[0][0]) :
                    template_text, _ = true_res[0][0]
                    template_text = template_text.strip().upper()
            
                if (true_res_thresh and true_res_thresh[0] and true_res_thresh[0][0]) :
                    template_text_thresh, _ = true_res_thresh[0][0]
                    template_text_thresh = template_text_thresh.strip().upper()
        
                if original_label != template_text and original_label != template_text_thresh:
                    self.defect_found = True
                    self.logger.log_main(f"Value Defect in {key} || Recognized_text: {template_text}  || Original_text: {original_label}", "error")
                    original_image = self.set_orientaion(true_image[y1:y2, x1:x2],orientaion)
                    defected_image = self.save_defect_image(original_image, test_crop, "Val_Error", key)
                    self.defects['defects'][0]["count"] +=1
                    self.defects['defects'][0]["data"].append({"Component": key, "image": defected_image})
                else:

                    if original_label == template_text:
                        self.logger.log_main(f"Inspection Done {key} || Recognized_text: {template_text}  || Original_text: {original_label}") 
                    else:
                        self.logger.log_main(f"Inspection Done {key} || Recognized_text: {template_text_thresh}  || Original_text: {original_label}")
                avg+= (time.time() - stime)
            
            except Exception as e:
                traceback_str = traceback.format_exc()
                self.logger.log_exception(f"Error in OCR component Reading: {key}: {traceback_str}")

        self.logger.log_time(f"OCR Inspection Average Time of each Component:  {round(avg/len( self.ocr_config.keys()) , 3)}secs")
            


    @torch.no_grad()
    def  __call__(
        self,
        test_image,
        true_image,
        pcb_id,
        pannel_id,
        start_time,
        **kwargs

    ):

        self.defects = {
            "Panel": f"Panel_{pannel_id}",
            "PCB": f"PCB_{pcb_id}",
            "defects":[
                {
                    "defect_type": "wrong_Values",
                    "count": 0,
                    "data": []
                },

                {
                    "defect_type": "missing",
                    "count": 0,
                    "data": []
                },

                {
                    "defect_type": "Shifted",
                    "count": 0,
                    "data": []
                },


                {
                "defect_type": "IC_Direction",
                "count": 0,
                "data": []
                },

                {
                "defect_type": "Bridge",
                "count": 0,
                "data": []
                },

                {   
                "defect_type": "Capacitor_Color",
                "count": 0,
                "data": []
                },

          ]
        }

        self.defect_found = False
        self.pcb_id = pcb_id
        self.pannel_id = pannel_id

        self.test_image = test_image
        self.true_image = true_image

        # Initializaing Template Configuration Files
        self.get_configurations()
        self.components_crops = {}
        self.component_ocr = {}

        keys_to_check = ["right_leg", "left_leg", "leg"]
        # Fteching Key and Values from JSON
        for key, box_data in self.template_config.items():
            present_values = {key: box_data[key] for key in keys_to_check if key in box_data}
            self.components_crops[key] = [box_data['value']] + list(present_values.values())
            self.component_ocr[key] = box_data['ocr_text']


        print("\n\n")

        # 1. OCR Reading Check
        self.logger.log_main(f"Starting Resistors Reading Inspection of PCB_{self.pcb_id}!")
        try:
            stime = time.time()
            self.ocr_inference(self.test_image, self.true_image)
            self.logger.log_time(f"OCR Inspection Time:  {round(time.time() - stime , 3)}secs")

        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.log_exception(f"Error in OCR component infernece: {traceback_str}")
        print("\n\n")

        # 2. model classifier
        self.logger.log_main(f"Starting Componenet Inspection of PCB_{self.pcb_id}!")
        try:
            stime = time.time()
            self.component_infernece(self.test_image, self.true_image)
            self.logger.log_time(f"Classifer Inspection Time:  {round(time.time() - stime , 3)}secs")
        except Exception as e:
            traceback_str = traceback.format_exc()
            self.logger.log_exception(f"Error in Classifier infernece: {traceback_str}")

        # detail json

        details = {
            "checklistItem": f"pcb_{pcb_id}",
            "points": "100%",
            "checkedItems": 85,
            "totalItems": round(time.time() - start_time , 3),
            "timeTaken": f"{round(time.time() - start_time , 3)} secs",
            "defectFound": "Defective" if self.defect_found else ""

        }

        print("\n\n")
        self.save_json(self.defects, "defects")
        self.save_json(details, "details")

        return self.defects
      