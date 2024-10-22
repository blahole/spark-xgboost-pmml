/*
修改spark的代码，使xgboost能够适应spark1.3版本
 */

package org.apache.spark

import java.net.URL

import org.apache.commons.logging.LogFactory
import org.apache.spark.scheduler.{SparkListener, SparkListenerExecutorRemoved, SparkListenerTaskEnd}
import org.codehaus.jackson.map.ObjectMapper

import scala.collection.JavaConverters._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future, TimeoutException}
import scala.util.control.ControlThrowable

/**
  * A tracker that ensures enough number of executor cores are alive.
  * Throws an exception when the number of alive cores is less than nWorkers.
  *
  * @param sc The SparkContext object
  * @param timeout The maximum time to wait for enough number of workers.
  * @param numWorkers nWorkers used in an XGBoost Job
  */
class SparkParallelismTracker(
                               val sc: SparkContext,
                               timeout: Long,
                               numWorkers: Int) {

  private[this] val requestedCores = numWorkers * sc.conf.getInt("org.apache.spark.task.cpus", 1)
  private[this] val mapper = new ObjectMapper()
  private[this] val logger = LogFactory.getLog("XGBoostSpark")
  private[this] val url = sc.uiWebUrl match {
    case Some(baseUrl) => new URL(s"$baseUrl/api/v1/applications/${sc.applicationId}/executors")
    case _ => null
  }

  private[spark] def numAliveCores: Int = {
    try {
      if (url != null) {
        mapper.readTree(url).findValues("totalCores").asScala.map(_.asInt).sum
      } else {
        Int.MaxValue
      }
    } catch {
      case ex: Throwable =>
        logger.warn(s"Unable to read total number of alive cores from REST API." +
          s"Health Check will be ignored.")
        ex.printStackTrace()
        Int.MaxValue
    }
  }

  private[spark] def waitForCondition(
                                      condition: => Boolean,
                                      timeout: Long,
                                      checkInterval: Long = 100L) = {
    val monitor = Future {
      while (!condition) {
        Thread.sleep(checkInterval)
      }
    }
    Await.ready(monitor, timeout.millis)
  }

  private[spark] def safeExecute[T](body: => T): T = {
    val listener = new TaskFailedListener
    sc.addSparkListener(listener)
    try {
      body
    } finally {
      //注销地方
//      sc.removeListener(listener)
      sc.listenerBus.removeListener(listener)
    }
  }

  /**
    * Execute a blocking function call with two checks on enough nWorkers:
    *  - Before the function starts, wait until there are enough executor cores.
    *  - During the execution, throws an exception if there is any executor lost.
    *
    * @param body A blocking function call
    * @tparam T Return type
    * @return The return of body
    */
  def execute[T](body: => T): T = {
    if (timeout <= 0) {
      logger.info("starting training without setting timeout for waiting for resources")
      body
    } else {
      try {
        logger.info(s"starting training with timeout set as $timeout ms for waiting for resources")
        waitForCondition(numAliveCores >= requestedCores, timeout)
      } catch {
        case _: TimeoutException =>
          throw new IllegalStateException(s"Unable to get $requestedCores workers for" +
            s" XGBoost training")
      }
      safeExecute(body)
    }
  }
}

private class ErrorInXGBoostTraining(msg: String) extends ControlThrowable {
  override def toString: String = s"ErrorInXGBoostTraining: $msg"
}

private[spark] class TaskFailedListener extends SparkListener {
  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    taskEnd.reason match {
      case taskEnd: SparkListenerTaskEnd =>
        if (taskEnd.reason.isInstanceOf[TaskFailedReason]) {
          throw new ErrorInXGBoostTraining(s"TaskFailed during XGBoost Training: " +
            s"${taskEnd.reason}")
        }
      case executorRemoved: SparkListenerExecutorRemoved =>
        throw new ErrorInXGBoostTraining(s"Executor lost during XGBoost Training: " +
          s"${executorRemoved.reason}")
      case _ =>
    }
  }
}