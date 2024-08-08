import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import time


import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_grad_cam import GradCAM
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.metrics.road import ROADMostRelevantFirst, ROADLeastRelevantFirst

""" This code is from the following repository: https://github.com/Cranial-XIX/CAGrad """
class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return iu, acc  # Changed from MIoU to IoU


def depth_error(x_pred, x_output, base_class_mask, extend=False):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
    x_pred_masked = x_pred.masked_select(binary_mask)
    x_label_masked = x_output.masked_select(binary_mask)

    # Convert base class mask to torch tensor
    base_class_mask_tensor = torch.tensor(base_class_mask, device=device)
    base_class_mask_tensor_masked = base_class_mask_tensor.masked_select(binary_mask)

    x_pred_class_mask = x_pred_masked.masked_select(base_class_mask_tensor_masked)
    x_label_class_mask = x_label_masked.masked_select(base_class_mask_tensor_masked)

    # Changed to class based depth errors
    abs_err_class = 0
    rel_err_class = 0
    count = 0
    
    for idx, pixel in enumerate(x_pred_class_mask):
        abs_err_class += torch.abs(pixel - x_label_class_mask[idx])
        rel_err_class += (torch.abs(pixel - x_label_class_mask[idx]) / x_label_class_mask[idx])
        count += 1

    return abs_err_class / count, rel_err_class / count


def normal_error(x_pred, x_output, base_class_mask):
    #x, y, z = signs
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0)

    # Convert base class mask to torch tensor
    base_class_mask_tensor = torch.tensor(base_class_mask, device=device)
    base_class_mask_tensor_masked = base_class_mask_tensor.masked_select(binary_mask)

    error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask).masked_select(base_class_mask_tensor_masked), -1, 1)).detach().cpu().numpy()
    error = np.degrees(error)
    return np.mean(error), np.median(error), np.mean(error < 11.25), np.mean(error < 22.5), np.mean(error < 30)
    
""" End of CAGrad Code """

class SemanticSegmentationTarget:
    def __init__(self, category, masks, task_type="semantic"):
        self.category = category
        self.task_type = task_type
        if task_type == "multi":
            self.mask = [torch.from_numpy(mask).requires_grad_(True) for mask in masks]
        else: 
            self.mask = torch.from_numpy(masks).requires_grad_(True)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        if self.task_type == "semantic":
            outputs = model_output
            output = outputs[0]
            output = output.squeeze(0)
            if self.category == "all":
                return (output[:, :, :] * self.mask).sum()
            return (output[self.category, :, :] * self.mask).sum()
        elif self.task_type == "depth":
            outputs = model_output
            output = outputs[1]
            return (output.squeeze(0) * self.mask).sum()
        elif self.task_type == "normals":
            outputs = model_output
            output = outputs[2]
            return (output.squeeze(0) * self.mask).sum()
        elif self.task_type == "multi":
            sums = 0
            for i in range(3):
                sums += (model_output[i].squeeze(0) * self.mask[i]).sum()
            return sums
    
    def __str__(self):
        return f"Category: {self.category}, Mask shape: {self.mask.shape}"

def show_image(image_tensor, type):
    image_np = image_tensor.cpu().numpy()
    image_np = image_np.squeeze(0)

    if type == "SemSeg":
        class_predict = np.argmax(image_np, axis=0)
        color_map = plt.cm.get_cmap('jet', len(np.unique(class_predict)))
        color_image = color_map(class_predict)
        plt.imshow(color_image)
        plt.title("Semantic Segmentation")
        plt.colorbar()
        plt.axis('off')
        plt.show()

    elif type == "Original":
        image_np = image_np.transpose(1, 2, 0)
        plt.imshow(image_np)
        plt.title("Original Image")
        plt.axis('off')
        plt.show()

    elif type == "Depth":
        image_np = image_np.squeeze(0)
        sns.heatmap(image_np, cmap=sns.color_palette("Spectral_r", as_cmap=True))
        plt.title("Depth")
        plt.axis('off')
        plt.show()

    elif type == "SurNorm":
        x = np.arange(384)
        y = np.arange(288)
        X, Y = np.meshgrid(x, y)

        # Extract components of surface normals
        U = image_np[0]
        V = image_np[1]
        W = image_np[2]

        # Plot the 3D quiver plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, np.zeros_like(X), U, V, W, length=0.1, normalize=True)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Surface Normals')
        plt.show()
            

def show_mask(test_data, test_pred, cat_class):
    normalized_mask = torch.nn.functional.softmax(test_pred, dim=1).cpu()
    mask_one = normalized_mask[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    mask_one_uint8 = 255 * np.uint8(mask_one == cat_class)
    mask_one_float = np.float32(mask_one == cat_class)            

    repeated_mask = np.repeat(mask_one_uint8[:, :, None], 3, axis=-1)

    return mask_one_float, repeated_mask

def show_mask_depth_and_norms(test_data, test_pred, cat_class):
    mask_one = test_pred[:, :, cat_class]
    mask_one_uint8 = 255 * np.uint8(mask_one == 1)
    mask_one_float = np.float32(mask_one == 1)      

    repeated_mask = np.repeat(mask_one_uint8[:, :, None], 3, axis=-1)

    return mask_one_float, repeated_mask

def show_seg_grad_cam(multi_task_model, test_data, cat_class, mask_one_float, device, task_type="semantic"):
    if task_type == "semantic":
        layer = [multi_task_model.decoder_att[0][4], multi_task_model.conv_block_dec[4]]
    elif task_type == "depth":
        layer =[multi_task_model.decoder_att[1][4], multi_task_model.conv_block_dec[4]]
    elif task_type == "normals":
        layer = [multi_task_model.decoder_att[2][4], multi_task_model.conv_block_dec[4]]
    target_layers = layer
    targets = [SemanticSegmentationTarget(cat_class, mask_one_float, task_type)]
    test_data.requires_grad = True
    
    test_data_np = test_data[0].cpu().numpy().transpose(1, 2, 0)

    with torch.enable_grad():
        with GradCAM(model=multi_task_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=test_data.to(device), targets=targets)[0, :]
            cam_image = show_cam_on_image(test_data_np, grayscale_cam, use_rgb=True)

    return cam_image, grayscale_cam

def show_full_grad_cam_seg(multi_task_model, test_data):
    target_layers = [multi_task_model.decoder_block_att[4][2]]
    mask_one_float = np.float32(torch.ones_like(test_data[0, 0, :, :]))
    targets = [SemanticSegmentationTarget("all", mask_one_float)]
    test_data.requires_grad = True
    test_data_np = test_data[0].cpu().numpy().transpose(1, 2, 0)

    with torch.enable_grad():
        with GradCAM(model=multi_task_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=test_data, targets=targets)[0, :]
            cam_image = show_cam_on_image(test_data_np, grayscale_cam, use_rgb=True)

    return cam_image

def show_grad_cam_all_tasks(multi_task_model, test_data, test_pred_full):
    # multi_task_model.decoder_block_att[4][2]
    target_layers = [multi_task_model.decoder_block_att[4][2]]
    masks_one_float = []
    for pred in test_pred_full: 
        b, h, w, d = pred.shape
        mask_one_float = np.float32(np.ones((h,w,d)))
        masks_one_float.append(mask_one_float)
    targets = [SemanticSegmentationTarget("all", masks_one_float, "multi")]
    test_data.requires_grad = True
    test_data_np = test_data[0].cpu().numpy().transpose(1, 2, 0)

    with torch.enable_grad():
        with GradCAM(model=multi_task_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=test_data, targets=targets)[0, :]
            cam_image = show_cam_on_image(test_data_np, grayscale_cam, use_rgb=True)

    return cam_image

def save_images(image, name):
    if name == "original":
        image = Image.fromarray(np.uint8(image * 255))
        image.save("./xplanations/original_image.png")
    if "depth" in name:
        image = Image.fromarray(image)
        image.save("./xplanations/Depths/" + name + ".png")
    if "norms" in name:
        image = Image.fromarray(image)
        image.save("./xplanations/Surface Normals/" + name + ".png")
    if "seg" in name:
        image = Image.fromarray(image)
        image.save("./xplanations/Segmentations/" + name + ".png")
    if "all" in name:
        image = Image.fromarray(image)
        image.save("./xplanations/All Tasks/" + name + ".png")

def assign_octile(x, y, z, norms_one_hot, norms_octiles, i, j):
    if z < 0:
        if x < 0 and y < 0:
            norms_one_hot[i][j][0] = 1
            norms_octiles[i][j] = 0
        if x < 0 and y >= 0:
            norms_one_hot[i][j][1] = 1
            norms_octiles[i][j] = 1
        if x >= 0 and y < 0:
            norms_one_hot[i][j][2] = 1
            norms_octiles[i][j] = 2
        if x >= 0 and y >= 0:
            norms_one_hot[i][j][3] = 1
            norms_octiles[i][j] = 3
    if z >= 0:
        if x < 0 and y < 0:
            norms_one_hot[i][j][4] = 1
            norms_octiles[i][j] = 4
        if x < 0 and y >= 0:
            norms_one_hot[i][j][5] = 1
            norms_octiles[i][j] = 5
        if x >= 0 and y < 0:
            norms_one_hot[i][j][6] = 1
            norms_octiles[i][j] = 6
        if x >= 0 and y >= 0:
            norms_one_hot[i][j][7] = 1
            norms_octiles[i][j] = 7

    return norms_one_hot, norms_octiles

def show_outputs(test_data, test_pred):
    # Display original image
    show_image(test_data, "Original")
    
    # Display semantic segmentation prediction
    show_image(test_pred[0], "SemSeg")

    # Display depth estimation prediction
    show_image(test_pred[1], "Depth")

    # Display surface normal estimation prediction
    # show_image(test_pred[2], "SurNorm")

def preprocess_depth(test_pred_full):
    test_pred = test_pred_full[1]
    image_np_depth = test_pred.cpu().numpy().squeeze(0).squeeze(0)
    depths = []
    for row in image_np_depth:
        for column in row:
            depths.append(column)
    quintiles = np.quantile(depths, [0, 0.2, 0.4, 0.6, 0.8, 1])
    image_np_depth_dec = image_np_depth.copy()
    for i in range(len(image_np_depth_dec)):
        for j in range(len(image_np_depth_dec[i])):
            # set to which decile the value belongs
            set = False
            for n in range(1, len(quintiles)):
                if image_np_depth_dec[i][j] <= quintiles[n]:
                    image_np_depth_dec[i][j] = n
                    set = True
                    break 
    
    return image_np_depth_dec, quintiles

def preprocess_surface_normals(test_pred_full):
    image_np_normal = test_pred_full[2].cpu().numpy().squeeze(0).transpose(1, 2, 0)
    h, w, d = image_np_normal.shape
    norms_one_hot = np.zeros((h, w, 8))
    norms_octiles = np.zeros((h, w))
    coord_product = np.zeros((h, w))
    for i in range(len(image_np_normal)):
        for j in range(len(image_np_normal[i])):
            x, y, z = image_np_normal[i][j]
            # if i <= 90 and j <= 90 and i >= 70 and j >= 70:
            #     print(i, j, " --  ", x, y, z, " --  ", image_np_normal[i][j])
            coord_product[i][j] = x * y * z
            norms_one_hot, norms_octiles = assign_octile(x, y, z, norms_one_hot, norms_octiles, i, j)

    return norms_one_hot

def generate_ROAD_inputs(test_data, grayscale_cam, targets, model, cat_class):
    road_images_percentile_and_class = []
    grayscale_cam = grayscale_cam[np.newaxis, ...]
    for perc in [100, 80, 60, 40, 20]:     
        cam_metric = ROADMostRelevantFirst(percentile=perc)
        perturbation_visualizations = cam_metric(test_data, grayscale_cam, targets, model, return_visualization=True, return_diff=False)
        road_images_percentile_and_class.append((perturbation_visualizations, perc, cat_class))
    
    return road_images_percentile_and_class

def generate_explanations(task, multi_task_model, test_data, test_pred_full, image_np_depth_dec, norms_one_hot, k, device):
    if task == "multi":
        cam_image = show_grad_cam_all_tasks(multi_task_model, test_data, test_pred_full)
        save_images(cam_image, "cam_all_tasks_3:" + str(k))

    if task == "SemSeg":
        test_pred = test_pred_full[0]
        unique_classes = np.unique(test_pred.argmax(axis=1).detach().cpu().numpy())
        # org_image = test_data[0].cpu().numpy().transpose(1, 2, 0)
        # save_images(org_image, "original:" + str(k))

        # cam_image_whole = show_full_grad_cam_seg(multi_task_model, test_data)
        # save_images(cam_image_whole, "cam_image_whole")
        road_images_percentile_and_class = []
        prev_clas_time = time.time()
        for cat_class in unique_classes:
            mask_one_float, mask = show_mask(test_data, test_pred, cat_class)
            cam_image, grayscale_cam = show_seg_grad_cam(multi_task_model, test_data, cat_class, mask_one_float, device)

            create_cam_time = time.time()
            print("Time to create CAM: ", round(create_cam_time - prev_clas_time, 3), " seconds", end=" ")

            # Check if the grayscale cam is not all zeros (there is some activation in the class)
            if grayscale_cam.sum() > 0:
                targets = [SemanticSegmentationTarget(cat_class, mask_one_float, "semantic")]
                road_images_percentile_and_class.extend(generate_ROAD_inputs(test_data, grayscale_cam, targets, multi_task_model, cat_class))
            one_class_time = time.time()
            prev_clas_time = one_class_time
            print("Time to create ROAD inputs for class", cat_class, ":", round(one_class_time - create_cam_time, 3), " seconds")

        return road_images_percentile_and_class


    if task == "Depth":
        h, w = image_np_depth_dec.shape
        iles = np.unique(image_np_depth_dec)
        counts = np.zeros(len(iles))
        one_hot_depth = np.zeros((h, w, len(iles)))
        for i in range(h):
            for j in range(w):
                class_index = image_np_depth_dec[i, j] - 1  # Adjust index to start from 0
                counts[int(class_index)] += 1
                one_hot_depth[i][j][int(class_index)] = 1

        road_images_percentile_and_class = []
        prev_clas_time = time.time()
        for depth_class in range(len(iles)):
            if counts[depth_class] != 0:
                mask_one_float, mask = show_mask_depth_and_norms(test_data, one_hot_depth, depth_class)
                #save_images(mask, "mask_image_depth" + str(depth_class) + ":" + str(k))
                cam_image, grayscale_cam = show_seg_grad_cam(multi_task_model, test_data, depth_class, mask_one_float, device, task_type="depth")
                
                create_cam_time = time.time()
                print("Time to create CAM: ", round(create_cam_time - prev_clas_time, 3), " seconds", end=" ")

                targets = [SemanticSegmentationTarget(depth_class, mask_one_float, "depth")]
                if mask_one_float.sum() > 0:
                    road_images_percentile_and_class.extend(generate_ROAD_inputs(test_data, grayscale_cam, targets, multi_task_model, depth_class))
                    
                one_class_time = time.time()
                prev_clas_time = one_class_time
                print("Time to create ROAD inputs for class", depth_class, ":", round(one_class_time - create_cam_time, 3), " seconds")

        return road_images_percentile_and_class
    
    if task == "SurNorm":
        counts = np.zeros(8)
        road_images_percentile_and_class = []
        prev_clas_time = time.time()
        for norm_class in range(8):
            mask_one_float, mask = show_mask_depth_and_norms(test_data, norms_one_hot, norm_class)
            # save_images(mask, "mask_image_surface_norms_pre_normalization" + str(norm_class) + ":" + str(k))
            cam_image, grayscale_cam = show_seg_grad_cam(multi_task_model, test_data, norm_class, mask_one_float, device, task_type="normals")
            # save_images(cam_image, "cam_image_surface_norms_pre_normalization" + str(norm_class) + ":" + str(k))
            create_cam_time = time.time()
            print("Time to create CAM: ", round(create_cam_time - prev_clas_time, 3), " seconds", end=" ")

            targets = [SemanticSegmentationTarget(norm_class, mask_one_float, "normals")]
            if mask_one_float.sum() > 0 and grayscale_cam.sum() > 0:
                road_images_percentile_and_class.extend(generate_ROAD_inputs(test_data, grayscale_cam, targets, multi_task_model, norm_class))
                
            one_class_time = time.time()
            prev_clas_time = one_class_time
            print("Time to create ROAD inputs for class", norm_class, ":", round(one_class_time - create_cam_time, 3), " seconds")

        return road_images_percentile_and_class

"""
=========== Universal Multi-task Trainer ===========
"""

def multi_task_tester(test_loader, multi_task_model, device):
    tasks = ["SemSeg", "Depth", "SurNorm", "multi"]
    modes = ["Generation", "Evaluation"]
    task = tasks[2]
    tasks = ["SemSeg", "Depth", "SurNorm"]
    mode = modes[0]
    seg_classes = ["Bed", "Books", "Ceiling", "Chair", "Floor", "Furniture", "Objects", "Picture", "Sofa", "Table", "TV", "Wall", "Window"]
    surface_norm_classes = [(-1,-1,-1), (-1, 1, -1), (1, -1, -1), (1, 1, -1), (-1, -1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, 1)] # signs of x, y and z
    print("Creating Explanations...")
    begin = time.time()

    # evaluating test data
    multi_task_model.eval()
    road_images_results = [] # List of tuples (perc, cat_class, metric, input_number)
    torch.manual_seed(6277)


    with torch.no_grad():  # operations inside don't track history
        test_dataset = iter(test_loader)
        for k in range(200):
            batch_start = time.time()
            test_data, test_label, test_depth, test_normal = next(test_dataset)
            test_data, test_label = test_data.to(device), test_label.long().to(device)
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            test_pred_full, _ = multi_task_model(test_data)

            # show_outputs(test_data, test_pred_full)
            # show_image(test_pred_full[0], "SemSeg")

            for task in tasks:
                start_task = time.time()
                print("Time till start of task: ", round(start_task - begin, 3), " seconds")
                """ Preprocess the regresion outputs """
                image_np_depth_dec, norms_one_hot = None, None
                if task == "Depth":
                    image_np_depth_dec, deciles = preprocess_depth(test_pred_full)
                if task == "SurNorm":
                    norms_one_hot = preprocess_surface_normals(test_pred_full)
                    
                pre_pro_time = time.time()
                print("Time to pre-processing: ", round(pre_pro_time - start_task, 3), " seconds")

                """ Generate Explanations """
                road_images_generated = generate_explanations(task, multi_task_model, test_data, test_pred_full, image_np_depth_dec, norms_one_hot, k, device)
                # road_images_generated in the form (pertubated image, percentile, class category)
                if task == "SemSeg":
                    print("Segmentation ROAD Results")
                    seg_road_images_results = []
                    for road_image in road_images_generated:
                        conf_mat = ConfMatrix(len(seg_classes))
                        perturbation_visualizations, perc, cat_class = road_image
                        pred_out, _ = multi_task_model(perturbation_visualizations.to(device))
                        seg_output = pred_out[0]

                        # accumulate label prediction for every pixel in training images
                        conf_mat.update(seg_output.argmax(1).flatten(), test_label.flatten())
                        class_iou_tensor, acc_tensor = conf_mat.get_metrics()
                        class_iou = class_iou_tensor.cpu().numpy()
                        acc = acc_tensor.cpu().numpy()
                        class_iou_cat = class_iou[cat_class]

                        seg_road_images_results.append((perc, cat_class, class_iou_cat, acc, task, k))
                    print("Done Evaluating Segmentation")

                    road_images_results.extend(seg_road_images_results)


                if task == "Depth":
                    print("Depth ROAD Results")
                    depth_road_images_results = []
                    for road_image in road_images_generated:
                        perturbation_visualizations, perc, depth_class = road_image
                        pred_out, _ = multi_task_model(perturbation_visualizations.to(device))
                        depth_output = pred_out[1]
                        depth_output_np = depth_output.cpu().numpy().squeeze(0).squeeze(0)
                        depths = depth_output_np.flatten()
                        quintiles = np.quantile(depths, [0, 0.2, 0.4, 0.6, 0.8, 1])
                        if perc == 100:
                            base_class_mask = (depth_output_np > quintiles[depth_class]) & (depth_output_np <= quintiles[depth_class+1])
                        err_abs, err_rel = depth_error(depth_output, test_depth, base_class_mask)
                        depth_road_images_results.append((perc, depth_class, err_rel.item(), err_abs.item(), task, k))
                    print("Done Evaluating Depth")

                    road_images_results.extend(depth_road_images_results)


                if task == "SurNorm":
                    print("Surface Normal ROAD Results")
                    surface_normals_road_images_results = []
                    for road_image in road_images_generated:
                        perturbation_visualizations, perc, norm_class = road_image
                        pred_out, _ = multi_task_model(perturbation_visualizations.to(device))
                        norm_output = pred_out[2]
                        norm_output_np = norm_output.cpu().numpy().squeeze(0)
                        if perc == 100:
                            x, y, z = surface_norm_classes[norm_class]
                            base_class_mask = (norm_output_np[0] * x > 0) & (norm_output_np[1] * y > 0) & (norm_output_np[2] * z > 0)
                        mean_angle, med_angle, _, _, _ = normal_error(norm_output, test_normal, base_class_mask)
                        surface_normals_road_images_results.append((perc, norm_class, mean_angle, med_angle, task, k))
                    print("Done Evaluating Surface Normals")

                    road_images_results.extend(surface_normals_road_images_results)
            
            batch_end = time.time()
            print("Time for one batch: ", round(batch_end - batch_start, 3), " seconds")

        df = pd.DataFrame(road_images_results, columns=["ROAD Percentile", "Class", "Class Metric", "Extra Class Metric", "Task", "Image Number"])
        df.to_csv("Results/ROAD_Results.csv")

