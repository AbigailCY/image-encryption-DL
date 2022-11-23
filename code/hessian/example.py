def validate(valid_loader, model, epoch, cur_step, writer, config):
    if config.network in ['ResNet18', 'ResNet34']:
        dim_feature = 512
    if config.network == 'ResNet50':
        dim_feature = 2048
    num_valid_sample = 358
    feature_map = torch.zeros(num_valid_sample, dim_feature).cuda()
    ground_truth = torch.zeros(num_valid_sample).cuda()
    index_begin = 0
    model.eval()
    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.cuda(), y.cuda()
            N = X.size(0)

            ft = model(X, y)

            index_end = index_begin + N
            feature_map[index_begin:index_end, :] = ft.view(N, dim_feature)
            ground_truth[index_begin:index_end] = y
            index_begin = index_end

        fm_n = feature_map.norm(p=2, dim=1)
        dist = 1 - torch.matmul(feature_map/fm_n.view(num_valid_sample, 1), (feature_map/fm_n.view(num_valid_sample, 1)).t())

        # metrics
        map = utils.map(ground_truth, dist)
        eer = utils.eer(ground_truth, dist)
        rank1 = utils.rankn(ground_truth, dist, 1)

        logger.info("Valid: [{:2d}/{}] Step {:03d}/{:03d}, EER {eer:.2%}, mAP {map:.2%}, Rank-1 {rank1:.2%}".format(
            epoch + 1, config.epochs, step + 1, len(valid_loader), eer=eer, map=map, rank1=rank1))

    if config.debug:
        writer.add_scalar('val/EER', eer, cur_step)
        writer.add_scalar('val/mAP', map, cur_step)
        writer.add_scalar('val/Rank-1', rank1, cur_step)

    return eer, map, rank1