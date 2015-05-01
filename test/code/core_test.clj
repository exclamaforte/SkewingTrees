(ns skewing.core-test
  (:require [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :as ct]
            [clojure.test :refer :all]
            [skewing.core :as core]
            [clojure.set]))

(ct/defspec sampled-elements-are-in-original-container
  50
  (prop/for-all [col (gen/container-type gen/any)
                 k gen/int]
                (clojure.set/subset? (into #{} (core/sample-with-replacement col k))
                                     (into #{} col))))
