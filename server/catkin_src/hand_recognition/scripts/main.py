#!/usr/bin/env python

from models.classifier import Classifier
from image_subscriber import ImageSubscriber
from hand_gesture import HandGestureRecognizer
import rospy


def main():
    rospy.init_node("hand_gesture_node", anonymous=False)

    model = Classifier().cuda()
    model.eval()

    img_subs = ImageSubscriber()
    model = HandGestureRecognizer()

    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        img = img_subs.wait_for_image()
        if img is None:
            continue

        prediction = model(img)

        rospy.loginfo("Prediction: {}".format(prediction))

        rate.sleep()


if __name__ == "__main__":
    main()